// OCS2 multiple-shooting SQP/HPIPM service for measured IsaacGym experiments.
// JSON lines on stdin/stdout keep ROS's C++ ABI separate from Conda/PyTorch.
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/constraint/LinearStateInputConstraint.h>
#include <ocs2_core/constraint/LinearStateConstraint.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_sqp/SqpSolver.h>
#include <chrono>
#include <iostream>

using namespace ocs2;
using json = nlohmann::json;

vector_t vec(const json& a) {
  auto v = a.get<std::vector<double>>();
  return Eigen::Map<vector_t>(v.data(), v.size());
}
matrix_t mat(const json& a) {
  matrix_t m(a.size(), a.at(0).size());
  for (int i = 0; i < m.rows(); ++i) m.row(i) = vec(a.at(i)).transpose();
  return m;
}
json array(const vector_t& v) { return std::vector<double>(v.data(), v.data() + v.size()); }

struct AffineMap final : SystemDynamicsBase {
  matrix_t A, B;
  vector_t c;
  AffineMap(const json& j) : A(mat(j.at("A"))), B(mat(j.at("B"))), c(vec(j.at("c"))) {
    // Explicit discrete identification, represented via Euler at exactly dt.
    // This is a grid map, not a continuous-time physical ODE approximation.
    const double dt = j.at("dt");
    A = (A - matrix_t::Identity(A.rows(), A.cols())).eval() / dt;
    B /= dt;
    c /= dt;
  }
  AffineMap* clone() const override { return new AffineMap(*this); }
  vector_t computeFlowMap(scalar_t, const vector_t& x, const vector_t& u, const PreComputation&) override {
    return A*x + B*u + c;
  }
  VectorFunctionLinearApproximation linearApproximation(scalar_t t, const vector_t& x, const vector_t& u,
                                                        const PreComputation& pc) override {
    VectorFunctionLinearApproximation f;
    f.f = computeFlowMap(t, x, u, pc); f.dfdx = A; f.dfdu = B;
    return f;
  }
};

struct Joint { Eigen::Isometry3d origin; Eigen::Vector3d axis; int index; };
struct PoseCost {
  std::vector<Joint> joints;
  std::vector<vector_t> references;
  vector_t weights;
  double roll, dt;
  int nx;
  explicit PoseCost(const json& j) : weights(vec(j.at("ee_weights"))), roll(j.at("roll")),
                                    dt(j.at("dt")), nx(j.at("state").size()) {
    for (const auto& q : j.at("chain")) {
      Joint joint;
      joint.origin = Eigen::Isometry3d::Identity();
      joint.origin.matrix() = mat(q.at("origin"));
      joint.axis = vec(q.at("axis"));
      joint.index = q.at("index");
      joints.push_back(joint);
    }
    for (const auto& r : j.at("references")) references.push_back(vec(r));
  }
  Eigen::Isometry3d pose(const vector_t& x) const {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = x.head<3>();
    T.linear() = (Eigen::AngleAxisd(x[3], Eigen::Vector3d::UnitZ()) *
                  Eigen::AngleAxisd(x[4], Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
    for (const auto& joint : joints) {
      T = T * joint.origin;
      if (joint.index >= 0) T.rotate(Eigen::AngleAxisd(x[8 + joint.index], joint.axis));
    }
    return T;
  }
  vector_t residual(double t, const vector_t& x) const {
    const auto& ref = references.at(std::min(references.size()-1,
                                static_cast<size_t>(std::llround(std::max(t, 0.) / dt))));
    const auto T = pose(x);
    Eigen::Quaterniond desired(ref[6], ref[3], ref[4], ref[5]);
    Eigen::AngleAxisd error(T.linear() * desired.normalized().toRotationMatrix().transpose());
    vector_t r(6);
    r.head<3>() = T.translation() - ref.head<3>();
    r.tail<3>() = error.angle() * error.axis();
    return r;
  }
  ScalarFunctionQuadraticApproximation quadratic(double t, const vector_t& x, int nu) const {
    auto cost = ScalarFunctionQuadraticApproximation::Zero(nx, nu);
    const auto r = residual(t, x);
    matrix_t J = matrix_t::Zero(6, nx);
    for (int k = 0; k < 14; ++k) {
      if (k >= 5 && k < 8) continue;
      vector_t xp=x, xm=x;
      xp[k] += 1e-5; xm[k] -= 1e-5;
      J.col(k) = (residual(t, xp) - residual(t, xm)) / 2e-5;
    }
    cost.f = .5 * r.dot(weights.cwiseProduct(r));
    cost.dfdx = J.transpose() * weights.cwiseProduct(r);
    cost.dfdxx = J.transpose() * weights.asDiagonal() * J;
    return cost;
  }
};

struct RunningCost final : StateInputCost {
  PoseCost pose;
  vector_t R, D;
  int previousIndex;
  explicit RunningCost(const json& j) : pose(j), R(vec(j.at("R"))), D(vec(j.at("D"))),
                                       previousIndex(j.at("previous_input_index")) {}
  RunningCost* clone() const override { return new RunningCost(*this); }
  scalar_t getValue(scalar_t t, const vector_t& x, const vector_t& u, const TargetTrajectories&,
                    const PreComputation&) const override {
    const auto r = pose.residual(t,x);
    const vector_t d = u - x.segment(previousIndex, u.size());
    return .5 * (r.dot(pose.weights.cwiseProduct(r)) + u.dot(R.cwiseProduct(u)) + d.dot(D.cwiseProduct(d)));
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t t, const vector_t& x, const vector_t& u,
                                            const TargetTrajectories&, const PreComputation&) const override {
    auto c = pose.quadratic(t,x,u.size());
    const int n = u.size(), p = previousIndex;
    const vector_t d = u - x.segment(p,n);
    c.f += .5 * (u.dot(R.cwiseProduct(u)) + d.dot(D.cwiseProduct(d)));
    c.dfdu = R.cwiseProduct(u) + D.cwiseProduct(d);
    c.dfduu = (R+D).asDiagonal();
    c.dfdx.segment(p,n) -= D.cwiseProduct(d);
    c.dfdxx.block(p,p,n,n) += D.asDiagonal();
    c.dfdux.block(0,p,n,n) = -D.asDiagonal().toDenseMatrix();
    return c;
  }
};
struct FinalCost final : StateCost {
  PoseCost pose;
  double weight;
  explicit FinalCost(const json& j) : pose(j), weight(j.at("terminal_weight")) {}
  FinalCost* clone() const override { return new FinalCost(*this); }
  scalar_t getValue(scalar_t t, const vector_t& x, const TargetTrajectories&, const PreComputation&) const override {
    const auto r = pose.residual(t,x);
    return .5 * weight * r.dot(pose.weights.cwiseProduct(r));
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t t, const vector_t& x,
                                    const TargetTrajectories&, const PreComputation&) const override {
    auto c = pose.quadratic(t,x,0);
    c.f *= weight; c.dfdx *= weight; c.dfdxx *= weight;
    return c;
  }
};

json solve(const json& j) {
  const auto start = std::chrono::steady_clock::now();
  const vector_t x = vec(j.at("state"));
  const matrix_t C = mat(j.at("constraint_C")), D = mat(j.at("constraint_D"));
  const vector_t e = vec(j.at("constraint_e"));
  const matrix_t F = mat(j.at("state_constraint_F"));
  const vector_t h = vec(j.at("state_constraint_h"));
  const int nu = D.cols();
  if (!x.allFinite() || C.cols()!=x.size() || D.rows()!=e.size() || C.rows()!=e.size())
    throw std::runtime_error("Invalid protocol dimensions or nonfinite initial state");
  OptimalControlProblem problem;
  problem.dynamicsPtr = std::make_unique<AffineMap>(j);
  problem.costPtr->add("world_ee_and_input", std::make_unique<RunningCost>(j));
  problem.finalCostPtr->add("world_ee", std::make_unique<FinalCost>(j));
  problem.inequalityConstraintPtr->add("commands_slew_target", std::make_unique<LinearStateInputConstraint>(e,C,D));
  problem.stateInequalityConstraintPtr->add("arm_target", std::make_unique<LinearStateConstraint>(h,F));
  problem.finalInequalityConstraintPtr->add("arm_target", std::make_unique<LinearStateConstraint>(h,F));
  // This reference SQP version logs bare inequalities but its HPIPM solve only
  // receives equalities. Use the same explicit barrier cost as its manipulator
  // example, and independently reject violations before applying any command.
  auto penalty = []() { return std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(.001, 1e-6)); };
  problem.softConstraintPtr->add("command_barrier", std::make_unique<StateInputSoftConstraint>(
      std::make_unique<LinearStateInputConstraint>(e,C,D), penalty()));
  problem.stateSoftConstraintPtr->add("target_barrier", std::make_unique<StateSoftConstraint>(
      std::make_unique<LinearStateConstraint>(h,F), penalty()));
  problem.finalSoftConstraintPtr->add("target_barrier", std::make_unique<StateSoftConstraint>(
      std::make_unique<LinearStateConstraint>(h,F), penalty()));
  DefaultInitializer initializer(nu);
  sqp::Settings settings;
  settings.dt = j.at("dt");
  settings.integratorType = SensitivityIntegratorType::EULER;
  settings.sqpIteration = j.value("iterations", 120);
  settings.nThreads = 1;
  settings.threadPriority = 0;
  settings.useFeedbackPolicy = false;
  settings.enableLogging = false;
  settings.inequalityConstraintMu = 0.;
  settings.deltaTol = 1e-7;
  settings.costTol = 1e-8;
  SqpSolver solver(settings, problem, initializer);
  const double horizon = j.at("horizon").get<int>() * settings.dt;
  solver.run(0., x, horizon);
  const auto sol = solver.primalSolution(horizon);
  const auto p = solver.getPerformanceIndeces();
  double margin = 1e100, defect = 0.;
  const matrix_t A = mat(j.at("A")), B = mat(j.at("B"));
  const vector_t c = vec(j.at("c"));
  for (size_t i=0; i<sol.stateTrajectory_.size(); ++i) {
    if (!sol.stateTrajectory_[i].allFinite()) throw std::runtime_error("Nonfinite optimized state");
    margin = std::min(margin, (F*sol.stateTrajectory_[i]+h).minCoeff());
    if (i+1<sol.stateTrajectory_.size()) {
      const auto& u = sol.inputTrajectory_.at(i);
      if (!u.allFinite()) throw std::runtime_error("Nonfinite optimized input");
      margin = std::min(margin, (C*sol.stateTrajectory_[i]+D*u+e).minCoeff());
      defect = std::max(defect, (sol.stateTrajectory_[i+1]-A*sol.stateTrajectory_[i]-B*u-c).lpNorm<Eigen::Infinity>());
      if (std::abs(sol.timeTrajectory_[i+1]-sol.timeTrajectory_[i]-settings.dt)>1e-7)
        throw std::runtime_error("Discrete model requires a uniform, event-free grid");
    }
  }
  const bool ok = margin > -1e-5 && defect < 1e-5;
  json out = {{"ok",ok}, {"solver","OCS2 SqpSolver / HPIPM"}, {"command",array(sol.inputTrajectory_.front())},
      {"predicted_state",array(sol.stateTrajectory_.at(1))}, {"constraint_margin",margin},
      {"max_dynamics_defect",defect}, {"dynamics_sse",p.dynamicsViolationSSE},
      {"inequality_sse",p.inequalityConstraintsSSE}, {"cost",p.cost}, {"iterations",solver.getNumIterations()},
      {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
  const auto T = PoseCost(j).pose(x);
  Eigen::Quaterniond quat(T.linear());
  vector_t ee(7); ee << T.translation(), quat.x(), quat.y(), quat.z(), quat.w();
  out["fk_at_state"] = array(ee);
  return out;
}

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    try { std::cout << solve(json::parse(line)).dump() << std::endl; }
    catch (const std::exception& e) {
      std::cout << json({{"ok",false},{"solver","OCS2 SqpSolver / HPIPM"},{"error",e.what()}}).dump() << std::endl;
    }
  }
}

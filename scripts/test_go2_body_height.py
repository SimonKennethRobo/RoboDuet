#!/usr/bin/env python3
"""Compute the flat-body Go2 standing-height range directly from its URDF.

No Isaac Gym environment or learned policy is loaded.  The script fixes the
base orientation to roll = pitch = 0, constrains all four foot-contact spheres
to the same horizontal ground plane, and finds the intersection of the four
legs' vertical reachability intervals under the URDF joint limits.
"""

import argparse
import itertools
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_URDF = "resources/robots/go2/urdf/arx5go2.urdf"
DEFAULT_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0


@dataclass
class FootContact:
    link: str
    center_offset: np.ndarray
    radius: float


@dataclass
class LegChain:
    foot: FootContact
    joints: List[Joint]
    actuated: List[Joint]


def _vector(value: Optional[str], default: Sequence[float]) -> np.ndarray:
    return np.asarray([float(item) for item in (value or " ".join(map(str, default))).split()], dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        return np.eye(3)
    x, y, z = axis / axis_norm
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _load_urdf(path: str) -> Tuple[Dict[str, Joint], Dict[str, FootContact]]:
    root = ET.parse(path).getroot()
    joints = {}
    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        joint_type = element.attrib["type"]
        if joint_type in ("revolute", "continuous"):
            lower = float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else -math.pi
            upper = float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else math.pi
        elif joint_type == "prismatic":
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        else:
            lower = upper = 0.0
        joint = Joint(
            name=element.attrib["name"],
            joint_type=joint_type,
            parent=element.find("parent").attrib["link"],
            child=element.find("child").attrib["link"],
            xyz=_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
            rpy=_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
            axis=_vector(axis.attrib.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            lower=lower,
            upper=upper,
        )
        if joint.child in joints:
            raise ValueError("URDF has multiple parent joints for link '{}'".format(joint.child))
        joints[joint.child] = joint

    contacts = {}
    for link in root.findall("link"):
        collision = link.find("collision")
        sphere = collision.find("geometry/sphere") if collision is not None else None
        if sphere is None:
            continue
        origin = collision.find("origin")
        contacts[link.attrib["name"]] = FootContact(
            link=link.attrib["name"],
            center_offset=_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
            radius=float(sphere.attrib["radius"]),
        )
    return joints, contacts


def _make_chain(foot: FootContact, child_joints: Dict[str, Joint], base_link: str) -> LegChain:
    chain = []
    current_link = foot.link
    while current_link != base_link:
        if current_link not in child_joints:
            raise ValueError("No joint chain from '{}' to base link '{}'".format(foot.link, base_link))
        joint = child_joints[current_link]
        chain.append(joint)
        current_link = joint.parent
    chain.reverse()
    actuated = [joint for joint in chain if joint.joint_type in ("revolute", "continuous", "prismatic")]
    if not actuated:
        raise ValueError("Foot '{}' has no actuated joints".format(foot.link))
    return LegChain(foot=foot, joints=chain, actuated=actuated)


def _contact_position(chain: LegChain, q: Sequence[float]) -> np.ndarray:
    if len(q) != len(chain.actuated):
        raise ValueError("Expected {} joint angles, got {}".format(len(chain.actuated), len(q)))
    transform = np.eye(4)
    q_iter = iter(q)
    for joint in chain.joints:
        transform = transform.dot(_transform(_rpy_matrix(joint.rpy), joint.xyz))
        if joint.joint_type in ("revolute", "continuous"):
            transform = transform.dot(_transform(_axis_angle_matrix(joint.axis, next(q_iter)), np.zeros(3)))
        elif joint.joint_type == "prismatic":
            transform = transform.dot(_transform(np.eye(3), joint.axis * next(q_iter)))
    sphere_center = transform[:3, :3].dot(chain.foot.center_offset) + transform[:3, 3]
    # The lowest point of the contact sphere touches the z=0 plane.
    return sphere_center + np.array([0.0, 0.0, -chain.foot.radius])


def _height(chain: LegChain, q: Sequence[float]) -> float:
    """Base height needed to place this foot's contact point on z=0."""
    return -float(_contact_position(chain, q)[2])


def _candidate_grid(bounds: Sequence[Tuple[float, float]], samples: int) -> Iterable[np.ndarray]:
    axes = [np.linspace(lower, upper, samples) for lower, upper in bounds]
    for candidate in itertools.product(*axes):
        yield np.asarray(candidate, dtype=float)


def _pattern_refine(chain: LegChain, seed: np.ndarray, direction: float, bounds, grid_step: np.ndarray) -> Tuple[float, np.ndarray]:
    """Refine a coarse-grid extremum with bounded coordinate pattern search."""
    current = seed.copy()
    current_value = _height(chain, current)
    step = grid_step.copy()
    for _ in range(24):
        improved = False
        for index in range(len(current)):
            for sign in (-1.0, 1.0):
                trial = current.copy()
                trial[index] = np.clip(trial[index] + sign * step[index], bounds[index][0], bounds[index][1])
                trial_value = _height(chain, trial)
                if direction * trial_value > direction * current_value:
                    current, current_value, improved = trial, trial_value, True
        if not improved:
            step *= 0.5
        if float(np.max(step)) < 1.0e-6:
            break
    return current_value, current


def _extreme_height(chain: LegChain, samples: int, direction: float) -> Tuple[float, np.ndarray]:
    bounds = [(joint.lower, joint.upper) for joint in chain.actuated]
    best = []
    for candidate in _candidate_grid(bounds, samples):
        value = _height(chain, candidate)
        best.append((direction * value, value, candidate))
    best.sort(key=lambda item: item[0], reverse=True)
    grid_step = np.asarray([(upper - lower) / (samples - 1) for lower, upper in bounds])
    refined = [_pattern_refine(chain, item[2], direction, bounds, grid_step) for item in best[:8]]
    return max(refined, key=lambda item: direction * item[0])


def _solve_height(chain: LegChain, target: float, samples: int) -> Tuple[float, np.ndarray]:
    """Find one joint-limit-valid stance with the requested body height."""
    bounds = [(joint.lower, joint.upper) for joint in chain.actuated]
    center = np.asarray([(lower + upper) * 0.5 for lower, upper in bounds])
    candidates = []
    for candidate in _candidate_grid(bounds, samples):
        candidates.append((abs(_height(chain, candidate) - target), candidate))
    candidates.sort(key=lambda item: item[0])
    initial_step = np.asarray([(upper - lower) / (samples - 1) for lower, upper in bounds])
    best_error = float("inf")
    best_q = None
    for _, seed in candidates[:8]:
        current = seed.copy()
        current_error = abs(_height(chain, current) - target)
        step = initial_step.copy()
        for _ in range(28):
            improved = False
            for index in range(len(current)):
                for sign in (-1.0, 1.0):
                    trial = current.copy()
                    trial[index] = np.clip(trial[index] + sign * step[index], bounds[index][0], bounds[index][1])
                    trial_error = abs(_height(chain, trial) - target)
                    if trial_error < current_error:
                        current, current_error, improved = trial, trial_error, True
            if not improved:
                step *= 0.5
            if float(np.max(step)) < 1.0e-6:
                break
        # Prefer a less-extreme solution if it reaches the same height.
        score = current_error + 1.0e-8 * float(np.sum((current - center) ** 2))
        if score < best_error:
            best_error, best_q = score, current
    return _height(chain, best_q), best_q


def _endpoint_solution(chains: Dict[str, LegChain], height: float, samples: int) -> Dict:
    legs = {}
    for name, chain in chains.items():
        reached_height, q = _solve_height(chain, height, samples)
        contact_in_base = _contact_position(chain, q)
        legs[name] = {
            "reached_body_height_m": reached_height,
            "height_error_m": reached_height - height,
            "joint_angles_rad": {joint.name: float(value) for joint, value in zip(chain.actuated, q)},
            "contact_point_in_base_m": [float(value) for value in contact_in_base],
            "contact_point_in_world_m": [float(contact_in_base[0]), float(contact_in_base[1]), float(height + contact_in_base[2])],
        }
    return legs


def main(args: argparse.Namespace) -> None:
    urdf_path = os.path.abspath(args.urdf)
    child_joints, contacts = _load_urdf(urdf_path)
    feet = tuple(item.strip() for item in args.feet.split(",") if item.strip())
    missing = [foot for foot in feet if foot not in contacts]
    if missing:
        raise ValueError("No spherical collision contact found for: {}".format(", ".join(missing)))

    chains = {foot: _make_chain(contacts[foot], child_joints, args.base_link) for foot in feet}
    per_leg = {}
    for foot, chain in chains.items():
        min_height, min_q = _extreme_height(chain, args.grid_samples, direction=-1.0)
        max_height, max_q = _extreme_height(chain, args.grid_samples, direction=1.0)
        per_leg[foot] = {
            "min_body_height_m": min_height,
            "max_body_height_m": max_height,
            "min_height_joint_angles_rad": {joint.name: float(value) for joint, value in zip(chain.actuated, min_q)},
            "max_height_joint_angles_rad": {joint.name: float(value) for joint, value in zip(chain.actuated, max_q)},
        }

    common_min = max(result["min_body_height_m"] for result in per_leg.values())
    common_max = min(result["max_body_height_m"] for result in per_leg.values())
    if common_min > common_max:
        raise RuntimeError("The four legs have no common flat-ground height range under their joint limits.")
    usable_min = max(common_min, args.min_body_height)
    if usable_min > common_max:
        raise RuntimeError(
            "--min-body-height {:.4f} m exceeds the kinematic maximum {:.4f} m.".format(
                args.min_body_height, common_max
            )
        )

    report = {
        "urdf": urdf_path,
        "base_link": args.base_link,
        "assumptions": {
            "base_roll_rad": 0.0,
            "base_pitch_rad": 0.0,
            "ground_plane_z_m": 0.0,
            "feet": list(feet),
            "contact_model": "lowest point of each foot collision sphere",
            "excluded": ["self-collision", "link-ground collision", "torque limits", "static stability margin", "dynamics"],
        },
        "grid_samples_per_joint": args.grid_samples,
        "per_leg": per_leg,
        "raw_joint_limit_body_height_range_m": {"min": common_min, "max": common_max, "span": common_max - common_min},
        "reported_body_height_range_m": {"min": usable_min, "max": common_max, "span": common_max - usable_min},
        "endpoint_joint_solutions": {
            "min_height": _endpoint_solution(chains, usable_min, args.grid_samples),
            "max_height": _endpoint_solution(chains, common_max, args.grid_samples),
        },
    }

    raw_common = report["raw_joint_limit_body_height_range_m"]
    common = report["reported_body_height_range_m"]
    print("URDF: {}".format(urdf_path))
    print("Assumption: roll=pitch=0, all four foot spheres contact z=0.")
    for foot in feet:
        result = per_leg[foot]
        print("{}: [{:.4f}, {:.4f}] m".format(foot, result["min_body_height_m"], result["max_body_height_m"]))
    print("Raw joint-limit range: [{:.4f}, {:.4f}] m".format(raw_common["min"], raw_common["max"]))
    print("Reported body-height range: [{:.4f}, {:.4f}] m (span {:.4f} m)".format(common["min"], common["max"], common["span"]))

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print("Detailed endpoint IK solutions: {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Go2 URDF path.")
    parser.add_argument("--base-link", default="base", help="URDF link used as the body-height reference.")
    parser.add_argument("--feet", default=",".join(DEFAULT_FEET), help="Comma-separated foot-link names.")
    parser.add_argument("--grid-samples", type=int, default=25, help="Coarse samples per actuated leg joint (>= 3).")
    parser.add_argument(
        "--min-body-height",
        type=float,
        default=0.0,
        help="Lower bound reported from the joint-limit range; default excludes nonphysical negative base heights.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parsed_args = parser.parse_args()
    if parsed_args.grid_samples < 3:
        parser.error("--grid-samples must be at least 3")
    if parsed_args.min_body_height < 0.0:
        parser.error("--min-body-height must be nonnegative")
    main(parsed_args)

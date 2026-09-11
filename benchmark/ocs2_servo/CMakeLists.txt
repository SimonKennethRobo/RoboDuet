cmake_minimum_required(VERSION 3.16)
project(roboduet_servo_ocs2 LANGUAGES CXX)
find_package(ocs2_sqp REQUIRED)
find_package(nlohmann_json REQUIRED)
add_executable(servo_ocs2 solver.cpp)
target_compile_features(servo_ocs2 PRIVATE cxx_std_17)
target_link_libraries(servo_ocs2 PRIVATE ocs2_sqp::ocs2_sqp nlohmann_json::nlohmann_json)
target_compile_options(servo_ocs2 PRIVATE -Wall -Wextra -Wno-unused-parameter)
# The reference hpipm_colcon library does not carry a RUNPATH for libhpipm.
# Transitive RPATH makes this standalone executable runnable outside sourced ROS.
target_link_options(servo_ocs2 PRIVATE -Wl,--disable-new-dtags)

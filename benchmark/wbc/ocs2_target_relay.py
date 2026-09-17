#!/usr/bin/python3
"""Relay benchmark SE(3) targets from ZeroMQ to OCS2's ROS topic."""

import argparse
import math

import rclpy
import zmq
from ocs2_msgs.msg import MpcInput, MpcState, MpcTargetTrajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("roboduet_benchmark_target_relay")
    publisher = node.create_publisher(
        MpcTargetTrajectories, "/mobile_manipulator_mpc_target", 1
    )
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(args.endpoint)
    try:
        while rclpy.ok():
            if socket.poll(timeout=10, flags=zmq.POLLIN):
                payload = socket.recv_json()
                target = [float(value) for value in payload["target"]]
                if len(target) != 7 or not all(math.isfinite(value) for value in target):
                    continue
                sim_time = float(payload["sim_time_s"])
                message = MpcTargetTrajectories()
                message.time_trajectory = [sim_time, sim_time + 1.0]
                for _ in range(2):
                    state = MpcState()
                    state.value = target
                    message.state_trajectory.append(state)
                    control = MpcInput()
                    control.value = [0.0] * 12
                    message.input_trajectory.append(control)
                publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        socket.close(linger=0)
        context.term()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Minimal test script for JoyController using ZMQ joystick input."""

import os
import argparse
import time

from go1_gym.envs.automatic.joy_wrapper import JoyController


def _format_commands(ctrl, dog_cmds, arm_cmds):
  dog = {}
  arm = {}
  for name, info in ctrl._channels.items():
    target = info.get("target")
    idx = info.get("index")
    if target == "dog" and idx < len(dog_cmds):
      dog[name] = f"{float(dog_cmds[idx]):5.2f}"
    elif target == "arm" and idx < len(arm_cmds):
      arm[name] = f"{float(arm_cmds[idx]):5.2f}"
  return dog, arm


def main():
  parser = argparse.ArgumentParser(
    description="Test JoyController with ZMQ joystick input."
  )
  parser.add_argument(
    "--endpoint",
    default="ipc:///tmp/roboduet_joy.sock",
    help="ZMQ endpoint (default: ipc:///tmp/roboduet_joy.sock)",
  )
  parser.add_argument(
    "--topic",
    default="joy",
    help="ZMQ topic (default: joy)",
  )
  parser.add_argument(
    "--config",
    default=os.path.join("config", "joy_mapping.yaml"),
    help="Joy mapping config path (default: config/joy_mapping.yaml)",
  )
  parser.add_argument(
    "--print-rate",
    type=float,
    default=10.0,
    help="Status print rate in Hz (default: 10)",
  )
  parser.add_argument(
    "--timeout",
    type=float,
    default=0.0,
    help="Optional stop time in seconds (0 = run forever).",
  )
  args = parser.parse_args()

  if not os.path.exists(args.config):
    raise SystemExit(f"Config not found: {args.config}")

  ctrl = JoyController(args.config)
  try:
    dog_cmds = [0.0, 0.0, 0.0]
    arm_cmds = [0.0] * 6
    next_print = time.time()
    deadline = time.time() + args.timeout if args.timeout > 0.0 else None

    print("Waiting for ZMQ joystick messages...")
    try:
      while True:
        action = ctrl.resolve_commands(dog_cmds, arm_cmds)
        if action == "reset":
          print("reset requested")

        now = time.time()
        if deadline and now >= deadline:
          break
        if now >= next_print:
          next_print = now + 1.0 / max(args.print_rate, 0.1)
          dog, arm = _format_commands(ctrl, dog_cmds, arm_cmds)
          print("connected:", ctrl.connected)
          print("dog:", dog)
          print("arm:", arm)
        time.sleep(0.01)
    except KeyboardInterrupt:
      print("\nexiting")
  finally:
    pass


if __name__ == "__main__":
    main()

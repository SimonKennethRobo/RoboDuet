#!/usr/bin/env python3

"""Publish joystick data from /dev/input/js0 over a local ZMQ socket."""

import argparse
import json
import os
import select
import struct
import time

import zmq


def _open_joystick(path):
    return open(path, "rb", buffering=0)


def _read_event(js):
    data = js.read(8)
    if len(data) != 8:
        return None
    time_ms, value, event_type, number = struct.unpack("IhBB", data)
    return time_ms, value, event_type, number


def _normalize_axis(value):
    # Linux joystick axis values are int16 in [-32767, 32767]
    if value >= 0:
        return min(value / 32767.0, 1.0)
    return max(value / 32767.0, -1.0)


def _cleanup_ipc(endpoint):
    if not endpoint.startswith("ipc://"):
        return
    path = endpoint[len("ipc://") :]
    if path and os.path.exists(path):
        os.remove(path)


def main():
    parser = argparse.ArgumentParser(
        description="Publish joystick data from /dev/input/js0 over ZMQ."
    )
    parser.add_argument(
        "--device",
        default="/dev/input/js0",
        help="Joystick device path (default: /dev/input/js0)",
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
        "--print-rate",
        type=float,
        default=50.0,
        help="Status print rate in Hz (default: 5)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.device):
        raise SystemExit(f"Device not found: {args.device}")

    _cleanup_ipc(args.endpoint)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(args.endpoint)
    time.sleep(0.1)

    axes = []
    buttons = []
    next_print = time.time()

    print(f"Publishing joystick data on {args.endpoint} topic '{args.topic}'")
    js = _open_joystick(args.device)
    try:
        while True:
            rlist, _, _ = select.select([js], [], [], 0.1)
            if rlist:
                evt = _read_event(js)
                if evt is not None:
                    _, value, event_type, number = evt

                    # Mask out init flag (0x80), leaving 0x01 button or 0x02 axis
                    evt_type = event_type & 0x7F
                    if evt_type == 0x02:  # axis
                        if number >= len(axes):
                            axes.extend([0.0] * (number + 1 - len(axes)))
                        axes[number] = _normalize_axis(value)
                    elif evt_type == 0x01:  # button
                        if number >= len(buttons):
                            buttons.extend([0] * (number + 1 - len(buttons)))
                        buttons[number] = 1 if value else 0

                    payload = json.dumps({"axes": axes, "buttons": buttons})
                    sock.send_multipart(
                        [args.topic.encode("utf-8"), payload.encode("utf-8")]
                    )

            now = time.time()
            if now >= next_print:
                next_print = now + 1.0 / max(args.print_rate, 0.1)
                print("axes:", axes)
                print("buttons:", buttons)
    except KeyboardInterrupt:
        print("\nexiting")
    finally:
        js.close()
        sock.close(0)


if __name__ == "__main__":
    main()

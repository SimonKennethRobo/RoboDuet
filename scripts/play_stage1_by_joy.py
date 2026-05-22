"""Pure stage1 policy inference with JoyLink joystick control."""

from scripts.play_by_joy import parse_args, main


if __name__ == "__main__":
    args = parse_args()
    args.stage1_only = True
    main(args)

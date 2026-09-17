"""Node-local dependency, identity, ROS, storage, and EGL preflight."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--ocs2-task", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy
    import torch

    from benchmark.wbc.formal_mujoco import _sha256

    required = [args.scene, args.suite, args.ocs2_task]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    probe = args.output_root / f".formal_write_probe_{os.getpid()}"
    probe.write_text("ok\n")
    probe.unlink()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        frame = renderer.render()
    finally:
        renderer.close()
    stack_root = Path(os.environ.get("WBC_RL_MPC_ROOT", "/home/simon-nfs/Projects/Simon/wbc_rl_mpc"))
    install = Path(os.environ.get("WBC_ROS_INSTALL", str(stack_root / "ros2_ws/install")))
    stale_prefixes = []
    if install.is_dir():
        for path in install.rglob("*.sh"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if "/home/simon/" in text and "/home/simon-nfs/" not in text:
                stale_prefixes.append(str(path))
                if len(stale_prefixes) == 20:
                    break
    ros_required = [
        install / "setup.zsh",
        install / "go2_x5_ocs2_bridge/lib/go2_x5_ocs2_bridge/wbc_benchmark_sync",
        install / "ocs2_mobile_manipulator/lib/libocs2_mobile_manipulator.a",
    ]
    qm_install = Path(os.environ.get(
        "QM_CONTROL_ROS_INSTALL",
        str(stack_root / "baselines/mpc_baseline/qm_control_baseline/install_aligned"),
    ))
    ros_required.extend([
        qm_install / "setup.bash",
        qm_install / "go2_x5_whole_body_mpc/lib/go2_x5_whole_body_mpc/go2_x5_sqp_mpc_node",
    ])
    missing_ros = [str(path) for path in ros_required if not path.is_file()]
    disk = shutil.disk_usage(args.output_root)
    result = {
        "status": "ready" if not stale_prefixes and not missing_ros else "blocked",
        "python": os.path.realpath(os.sys.executable),
        "versions": {"mujoco": mujoco.__version__, "numpy": numpy.__version__, "torch": torch.__version__},
        "frozen_inputs": {str(path.resolve()): _sha256(path) for path in required},
        "egl_frame": {"shape": list(frame.shape), "finite": bool(numpy.isfinite(frame).all())},
        "ffmpeg": shutil.which("ffmpeg"), "imageio_ffmpeg": None,
        "ros_install": str(install.resolve()),
        "qm_ros_install": str(qm_install.resolve()),
        "output_free_bytes": disk.free, "stale_ros_prefix_files": stale_prefixes,
        "missing_ros_files": missing_ros,
    }
    try:
        import imageio_ffmpeg
        result["imageio_ffmpeg"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        result["imageio_ffmpeg_error"] = repr(error)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ready" and result["egl_frame"]["finite"] and result["imageio_ffmpeg"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

import time


DEFAULT_DOG_COMMAND_NAMES = {
    0: "x_vel",
    1: "y_vel",
    2: "yaw_vel",
    3: "body_pitch",
    4: "body_roll",
    5: "body_height_delta",
    6: "gait_freq",
    7: "footswing_height",
    8: "stance_width",
    9: "stance_length",
    10: "gait_duration",
}


def add_rerun_args(parser):
    parser.add_argument("--rerun", action="store_true", default=False, help="Enable rerun.io telemetry viewer.")
    parser.add_argument("--rerun_hz", type=float, default=20.0, help="Rerun telemetry logging frequency.")
    parser.add_argument(
        "--rerun_torque_joints",
        type=int,
        default=12,
        help="Number of leading joints whose torques are logged by default.",
    )
    parser.add_argument(
        "--rerun_window_seconds",
        type=float,
        default=10.0,
        help="Visible sliding time window for rerun time-series panes; <=0 shows all history.",
    )
    parser.add_argument(
        "--rerun_joint_state",
        action="store_true",
        default=False,
        help="Also log all joint positions and velocities to the joint torque pane.",
    )
    parser.add_argument(
        "--rerun_extra_panes",
        action="store_true",
        default=False,
        help="Also log full base and command panes; these are not created by default.",
    )
    return parser


def make_rerun_logger(args, *, dog_command_names=None, app_id="roboduet_play"):
    return RerunLogger(
        enabled=bool(getattr(args, "rerun", False)),
        app_id=app_id,
        log_hz=float(getattr(args, "rerun_hz", 20.0)),
        torque_joint_count=int(getattr(args, "rerun_torque_joints", 12)),
        log_joint_state=bool(getattr(args, "rerun_joint_state", False)),
        log_extra_panes=bool(getattr(args, "rerun_extra_panes", False)),
        window_seconds=float(getattr(args, "rerun_window_seconds", 10.0)),
        dog_command_names=dog_command_names,
    )


class RerunLogger:
    """Optional rerun.io telemetry for the first simulated robot."""

    def __init__(
        self,
        enabled=False,
        app_id="roboduet_play",
        log_hz=20.0,
        torque_joint_count=12,
        log_joint_state=False,
        log_extra_panes=False,
        window_seconds=10.0,
        dog_command_names=None,
    ):
        self.enabled = False
        self._rr = None
        self._dog_command_names = dict(dog_command_names or DEFAULT_DOG_COMMAND_NAMES)
        self._torque_joint_count = max(0, int(torque_joint_count))
        self._log_joint_state = bool(log_joint_state)
        self._log_extra_panes = bool(log_extra_panes)
        self._window_seconds = max(0.0, float(window_seconds))
        self._last_log_time = 0.0
        self._log_period = 1.0 / max(float(log_hz), 1e-6)
        self._start_time = time.monotonic()
        self._step = 0
        self._joint_names = None
        self._warned_log_error = False

        if not enabled:
            return

        try:
            import rerun as rr  # type: ignore
        except ImportError:
            print("[rerun] disabled: install with `python -m pip install rerun-sdk`", flush=True)
            return

        self._rr = rr
        try:
            rr.init(app_id)
            rr.spawn()
        except Exception as exc:  # rerun viewer launch/connect errors should not break play.
            print(f"[rerun] disabled: failed to start viewer: {exc}", flush=True)
            return

        self._send_blueprint()
        self.enabled = True
        print(
            f"[rerun] logging telemetry at {log_hz:.1f} Hz "
            f"(torque joints={self._torque_joint_count}, joint_state={self._log_joint_state}, "
            f"extra_panes={self._log_extra_panes}, window={self._window_seconds:.1f}s)",
            flush=True,
        )

    def _make_top_left_legend(self, rrb):
        if not hasattr(rrb, "PlotLegend"):
            return None
        for corner in ("LeftTop", "left_top", "top_left"):
            try:
                return rrb.PlotLegend(corner=corner, visible=True)
            except Exception:
                pass
        try:
            return rrb.PlotLegend(visible=True)
        except Exception:
            return None

    def _make_visible_time_ranges(self, rrb):
        if self._window_seconds <= 0.0:
            return None
        if not hasattr(rrb, "VisibleTimeRange") or not hasattr(rrb, "TimeRangeBoundary"):
            return None
        try:
            return [
                rrb.VisibleTimeRange(
                    "time",
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-self._window_seconds),
                    end=rrb.TimeRangeBoundary.cursor_relative(),
                )
            ]
        except Exception:
            return None

    def _send_blueprint(self):
        rr = self._rr
        if not hasattr(rr, "send_blueprint"):
            print("[rerun] blueprint unavailable; legend corner must be set manually in this rerun version", flush=True)
            return
        try:
            import rerun.blueprint as rrb  # type: ignore
        except Exception:
            print("[rerun] blueprint unavailable; legend corner must be set manually in this rerun version", flush=True)
            return
        if not hasattr(rrb, "TimeSeriesView") or not hasattr(rrb, "Blueprint"):
            print("[rerun] blueprint views unavailable; legend corner must be set manually in this rerun version", flush=True)
            return

        legend = self._make_top_left_legend(rrb)
        view_kwargs = {"plot_legend": legend} if legend is not None else {}
        time_ranges = self._make_visible_time_ranges(rrb)
        if time_ranges is not None:
            view_kwargs["time_ranges"] = time_ranges
        elif self._window_seconds > 0.0:
            print("[rerun] blueprint time window unavailable in this rerun version", flush=True)
        try:
            views = [
                rrb.TimeSeriesView(origin="/pane_vx", name="vx", **view_kwargs),
                rrb.TimeSeriesView(origin="/pane_vy", name="vy", **view_kwargs),
                rrb.TimeSeriesView(origin="/pane_pitch", name="pitch", **view_kwargs),
                rrb.TimeSeriesView(origin="/pane_joint_torque", name="joint torque", **view_kwargs),
            ]
            if self._log_extra_panes:
                views.extend(
                    [
                        rrb.TimeSeriesView(origin="/pane_base", name="base", visible=False, **view_kwargs),
                        rrb.TimeSeriesView(origin="/pane_command", name="command", visible=False, **view_kwargs),
                    ]
                )
            blueprint = rrb.Blueprint(*views, collapse_panels=True)
            rr.send_blueprint(blueprint)
        except Exception as exc:
            print(f"[rerun] blueprint unavailable; legend corner must be set manually: {exc}", flush=True)

    @staticmethod
    def _safe_name(name):
        return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(name))

    def _set_time(self, elapsed_time):
        rr = self._rr
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("step", self._step)
        elif hasattr(rr, "set_time_sequence_entry"):
            rr.set_time_sequence_entry("step", self._step)

        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("time", elapsed_time)
        elif hasattr(rr, "set_time_seconds_entry"):
            rr.set_time_seconds_entry("time", elapsed_time)

    def _log_scalar(self, path, value, label=None):
        value = float(value)
        rr = self._rr
        if hasattr(rr, "Scalar"):
            rr.log(path, rr.Scalar(value))
        elif hasattr(rr, "log_scalar"):
            kwargs = {"label": label} if label else {}
            rr.log_scalar(path, value, **kwargs)
        else:
            raise AttributeError("rerun module has neither Scalar nor log_scalar")

    def log(self, env):
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_log_time < self._log_period:
            return
        self._last_log_time = now
        self._step += 1

        base = env.env
        self._set_time(now - self._start_time)

        try:
            self._log_env(base, env)
        except Exception as exc:
            if not self._warned_log_error:
                print(f"[rerun] disabled: logging failed: {exc}", flush=True)
                self._warned_log_error = True
            self.enabled = False

    def _log_env(self, base, env):
        lin_vel = base.base_lin_vel[0].detach().cpu().tolist()
        ang_vel = base.base_ang_vel[0].detach().cpu().tolist()
        command = env.commands_dog[0].detach().cpu().tolist()
        torques = base.torques[0].detach().cpu().tolist()
        pitch = base.pitch[0].detach().cpu().item()
        roll = base.roll[0].detach().cpu().item()
        height = base.root_states[0, 2].detach().cpu().item()

        if len(command) > 0:
            self._log_scalar("pane_vx/base", lin_vel[0])
            self._log_scalar("pane_vx/command", command[0])
        if len(command) > 1:
            self._log_scalar("pane_vy/base", lin_vel[1])
            self._log_scalar("pane_vy/command", command[1])
        if len(command) > 3:
            self._log_scalar("pane_pitch/base", pitch)
            self._log_scalar("pane_pitch/command", command[3])

        if self._log_extra_panes:
            self._log_scalar("pane_base/lin_vel/x", lin_vel[0])
            self._log_scalar("pane_base/lin_vel/y", lin_vel[1])
            self._log_scalar("pane_base/lin_vel/z", lin_vel[2])
            self._log_scalar("pane_base/ang_vel/x", ang_vel[0])
            self._log_scalar("pane_base/ang_vel/y", ang_vel[1])
            self._log_scalar("pane_base/ang_vel/z", ang_vel[2])
            self._log_scalar("pane_base/pose/height", height)
            self._log_scalar("pane_base/pose/pitch", pitch)
            self._log_scalar("pane_base/pose/roll", roll)

            for idx, name in self._dog_command_names.items():
                if idx < len(command):
                    self._log_scalar(f"pane_command/dog/{name}", command[idx])

        if self._joint_names is None:
            raw_names = getattr(base, "dof_names", None) or [f"dof_{i}" for i in range(len(torques))]
            self._joint_names = [self._safe_name(name) for name in raw_names]

        torque_count = min(self._torque_joint_count, len(torques), len(self._joint_names))
        for idx, name in enumerate(self._joint_names[:torque_count]):
            line_name = f"{idx:02d}_{name}"
            self._log_scalar(f"pane_joint_torque/{line_name}", torques[idx], label=line_name)

        if self._log_joint_state:
            dof_vel = base.dof_vel[0].detach().cpu().tolist()
            dof_pos = base.dof_pos[0].detach().cpu().tolist()
            for idx, name in enumerate(self._joint_names[: len(torques)]):
                if idx < len(dof_vel):
                    self._log_scalar(f"pane_joint_torque/joint_state/velocity/{idx:02d}_{name}", dof_vel[idx])
                if idx < len(dof_pos):
                    self._log_scalar(f"pane_joint_torque/joint_state/position/{idx:02d}_{name}", dof_pos[idx])

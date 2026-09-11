"""Optional W&B telemetry; failures must not terminate training."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from go1_gym.envs.config import cfg_to_dict


class WandbLogger:
    def __init__(self, args, cfg, recipe, log_dir):
        self.run = None
        self.failures = 0
        self.environment_steps = 0
        if args.no_wandb:
            return
        try:
            import wandb

            now = datetime.now()
            group = args.run_name or Path(log_dir).name
            self.run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=group,
                name=f"{now:%Y-%m-%d}/{group}_{now:%H%M%S}",
                notes=getattr(args, "notes", ""),
                settings=wandb.Settings(console="off"),
                job_type=args.stage,
                tags=["ma2022", args.stage, f"seed{args.seed}"],
                mode="offline" if args.offline else "online",
                dir=str(Path(log_dir).resolve()),
                config={
                    "Cfg": cfg_to_dict(cfg),
                    "MaTrainingConfig": asdict(recipe),
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
            )
            self.run.define_metric("iteration")
            self.run.define_metric("*", step_metric="iteration")
        except Exception as error:
            self._warn(error)
            self.finish(exit_code=1)

    def _warn(self, error):
        self.failures += 1
        if self.failures == 1 or self.failures % 100 == 0:
            print(f"[wandb] Skipping telemetry after logging error: {error}", flush=True)

    def log(self, iteration, metrics, *, num_envs, rollout_steps, learning_rate, videos=()):
        if self.run is None:
            return
        self.environment_steps += num_envs * rollout_steps
        payload = {"iteration": iteration,
                   "Train/run_environment_steps": self.environment_steps,
                   "Train/learning_rate": learning_rate}
        for key, value in metrics.items():
            if key == "reward":
                name = "Train/reward"
            elif key == "loss" or key in ("action", "embedding", "privileged", "scan", "w1", "w2"):
                name = f"Loss/{key}"
            else:
                name = f"Performance/{key}"
            payload[name] = value
        try:
            if videos:
                import wandb
                payload["Video/training"] = [wandb.Video(str(path), format="mp4") for path in videos]
            self.run.log(payload, step=iteration)
        except Exception as error:
            self._warn(error)

    def finish(self, exit_code=0):
        run, self.run = self.run, None
        if run is not None:
            try:
                run.finish(exit_code=exit_code)
            except Exception as error:
                self._warn(error)

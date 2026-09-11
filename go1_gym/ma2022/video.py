"""Bounded training clips streamed to disk, including episode resets."""

import math
from pathlib import Path


class TrainingVideo:
    def __init__(self, directory, stage, dt, *, interval=500, seconds=10., stride=2):
        self.directory = Path(directory) / "videos"
        self.stage, self.interval, self.stride = stage, interval, stride
        steps = seconds / dt
        # PhysX exposes dt as float32: 0.2 / dt can be 10.0000002.
        # Avoid turning an exact-duration clip into one extra frame.
        self.max_steps = max(1, round(steps) if math.isclose(steps, round(steps), rel_tol=1e-6)
                             else math.ceil(steps))
        self.fps = 1. / (dt * stride)
        self.writer = None
        self.completed = []
        self.steps = self.frames = 0

    def start(self, iteration, first=False):
        if self.writer is not None or (not first and iteration % self.interval != 0):
            return
        try:
            import imageio.v2 as imageio

            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self.directory / f"{self.stage}_{iteration:06d}.mp4"
            self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264",
                                            pixelformat="yuv420p", macro_block_size=1)
            self.steps = self.frames = 0
        except Exception as error:
            print(f"[video] Skipping clip: {error}", flush=True)

    def capture(self, env):
        if self.writer is None:
            return
        try:
            if self.steps % self.stride == 0:
                from isaacgym import gymapi

                index = env._render_camera_env
                x, y, z = (float(v) for v in env.root_states[index, :3])
                env.gym.set_camera_location(env.rendering_camera, env.envs[index],
                                            gymapi.Vec3(x + 1., y - 1.5, z + .8),
                                            gymapi.Vec3(x, y, z))
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                rgba = env.gym.get_camera_image(env.sim, env.envs[index],
                                                env.rendering_camera, gymapi.IMAGE_COLOR)
                rgba = rgba.reshape(env.camera_props.height, env.camera_props.width, 4)
                self.writer.append_data(rgba[..., :3].copy())
                self.frames += 1
            self.steps += 1
            if self.steps >= self.max_steps:
                self.finish()
        except Exception as error:
            print(f"[video] Stopping failed clip: {error}", flush=True)
            self.finish(publish=False)

    def finish(self, publish=True):
        writer, self.writer = self.writer, None
        if writer is not None:
            try:
                writer.close()
                if publish and self.frames:
                    self.completed.append(self.path)
                    print(f"[video] Saved {self.path} ({self.frames} frames)", flush=True)
            except Exception as error:
                print(f"[video] Could not finalize clip: {error}", flush=True)

    def pop_completed(self):
        paths, self.completed = self.completed, []
        return paths

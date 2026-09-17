"""Quarantine invalid physics samples before terrain/reward/observation queries.

Only tensor bookkeeping happens here. The existing reset_idx path performs the
single ordered DOF/root reset at the end of the step. Invalid transitions are
terminal with zero task reward; valid environments are left untouched.
"""
from pathlib import Path
from collections import deque
import torch
from go1_gym.file_io import optional_output


def physics_context(env, ids):
    """Raw per-environment controller/reset context; performs no sim writes."""
    fields = ('root_states', 'dof_pos', 'dof_vel', 'actions', 'torques',
              'commands_dog', 'episode_length_buf', 'next_push_step',
              'stage1_arm_target_offset', 'stage1_arm_target_vel')
    return {key: getattr(env, key)[ids].detach().clone() for key in fields
            if isinstance(getattr(env, key, None), torch.Tensor)}


def record_physics_context(env):
    count = getattr(env.cfg.env, 'numerical_trace_steps', 0)
    if not count:
        return
    if not hasattr(env, '_numerical_trace'):
        env._numerical_trace = deque(maxlen=count)
    env._numerical_trace.append(dict(step=env.common_step_counter,
                                    values=physics_context(env, slice(0, env.num_envs))))


def fault_context(env, ids):
    """Select only failed rows when transferring the short GPU trace to CPU."""
    ids = ids[:8]
    history = [dict(step=frame['step'], values={k: v[ids].cpu() for k, v in frame['values'].items()})
               for frame in getattr(env, '_numerical_trace', ())]
    return dict(step=env.common_step_counter, ids=ids.cpu(), history=history,
                current={k: v.cpu() for k, v in physics_context(env, ids).items()},
                dof_names=getattr(env, 'dof_names', None),
                control_type=getattr(getattr(env.cfg, 'control', None), 'control_type', None))


def quarantine_physics(env):
    n = env.num_envs
    root = env.root_states[:n]
    rigid = env.rigid_body_state.view(n, env.num_bodies, 13)
    contacts = env.contact_forces.view(n, env.num_bodies, 3)
    bad = ~torch.isfinite(root).all(dim=1)
    for value in (env.dof_pos, env.dof_vel, rigid, contacts):
        bad |= ~torch.isfinite(value).reshape(n, -1).all(dim=1)
    qnorm = torch.linalg.vector_norm(root[:, 3:7], dim=1)
    bad |= (qnorm < 0.5) | (qnorm > 1.5)
    # Far outside attainable motion: catches finite PhysX explosions too.
    bad |= (root[:, 7:13].abs() > env.cfg.env.numerical_max_root_speed).any(dim=1)
    bad |= (env.dof_vel.abs() > env.cfg.env.numerical_max_dof_speed).any(dim=1)
    env.numerical_fault_mask.copy_(bad)
    ids = bad.nonzero(as_tuple=False).flatten()
    env.numerical_fault_active = ids.numel() > 0
    if ids.numel() == 0:
        return
    env.numerical_fault_count += ids.numel()
    print(f"[numerical fault] step={env.common_step_counter} count={ids.numel()} "
          f"total={env.numerical_fault_count} ids={ids[:8].tolist()}", flush=True)
    directory = getattr(env, 'numerical_fault_log_dir', None)
    if directory and env.numerical_fault_dumps < 3:
        chosen = ids[:8]
        with optional_output('numerical fault state'):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(dict(step=env.common_step_counter, ids=chosen.cpu(),
                            root=root[chosen].cpu(), dof_pos=env.dof_pos[chosen].cpu(),
                            dof_vel=env.dof_vel[chosen].cpu(), rigid=rigid[chosen].cpu(),
                            contacts=contacts[chosen].cpu(), actions=env.actions[chosen].cpu(),
                            commands=env.commands_dog[chosen].cpu(),
                            context=fault_context(env, chosen)),
                       path / f'fault-{env.numerical_fault_dumps}.pt')
        env.numerical_fault_dumps += 1

    # Finite placeholders for this rejected sample, not a replacement physical
    # trajectory. reset_idx below writes newly sampled valid states to PhysX.
    root[ids] = env.base_init_state
    root[ids, :3] += env.env_origins[ids]
    env.dof_pos[ids] = env.default_dof_pos
    env.dof_vel[ids] = 0
    rigid[ids] = 0
    rigid[ids, :, :3] = root[ids, None, :3]
    rigid[ids, :, 6] = 1
    contacts[ids] = 0
    env.step_locomotion_power[ids] = 0
    if hasattr(env, "step_locomotion_abs_energy_j"):
        env.step_locomotion_abs_energy_j[ids] = 0
    if hasattr(env, "step_locomotion_positive_energy_j"):
        env.step_locomotion_positive_energy_j[ids] = 0
    env.torques[ids] = 0
    env.actions[ids] = 0

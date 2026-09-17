import json

import mujoco
import numpy as np

from benchmark.wbc.formal_mujoco import (
    ENVIRONMENT_ID, materialize_environment, mild_rough_heightfield,
    select_shard_tasks,
)


def test_mild_rough_heightfield_is_seeded_bounded_and_flat_at_spawn():
    first = mild_rough_heightfield(17, 17, 0.01)
    second = mild_rough_heightfield(17, 17, 0.01)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (17, 17)
    assert first[8, 8] == 0.0
    assert np.max(np.abs(first)) <= 0.01 + 1e-12
    assert np.std(first) > 0.001


def test_materialized_environment_compiles_and_records_contract(tmp_path):
    scene, manifest = materialize_environment(
        tmp_path, seed=17, resolution=17, half_extent_m=8.0,
        peak_height_m=0.10, slide_friction=0.8,
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    floor = model.geom("floor").id
    assert manifest["environment_id"] == ENVIRONMENT_ID
    assert model.nhfield == 1
    assert model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert model.opt.timestep == 0.0025
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    np.testing.assert_allclose(model.geom_friction[floor], [0.8, 0.02, 0.01])
    assert manifest["method_specific_dynamics_allowed_for"] == ["umi"]
    assert manifest["method_specific_dynamics_exceptions"]["umi"]["profile"] == "training_nominal"
    assert json.loads((tmp_path / "environment.json").read_text()) == manifest


def test_task_shards_are_deterministic_complete_and_disjoint():
    tasks = [{"task_id": f"T{index}"} for index in range(68)]
    shards = [select_shard_tasks(tasks, 7, index) for index in range(7)]
    assert [len(shard) for shard in shards] == [10, 10, 10, 10, 10, 9, 9]
    ids = [task["task_id"] for shard in shards for task in shard]
    assert len(ids) == len(set(ids)) == 68
    assert set(ids) == {task["task_id"] for task in tasks}

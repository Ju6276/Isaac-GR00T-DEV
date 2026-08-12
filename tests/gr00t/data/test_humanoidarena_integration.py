import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "HumanoidArena"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_http_bridge_preserves_canonical_action_order():
    bridge = load_module("humanoidarena_http_bridge", "serve_http_bridge.py")
    horizon = 30
    actions = {}
    start = 0
    for key, dim in zip(bridge.ACTION_KEYS, bridge.ACTION_DIMS, strict=True):
        values = np.arange(start, start + dim, dtype=np.float32)
        actions[key] = np.tile(values, (1, horizon, 1))
        start += dim
    chunk = bridge.concatenate_action_chunk(actions)
    assert chunk.shape == (horizon, 40)
    np.testing.assert_array_equal(chunk[0], np.arange(40, dtype=np.float32))


def test_http_bridge_preserves_canonical_state_order():
    bridge = load_module("humanoidarena_http_bridge_state", "serve_http_bridge.py")
    state = np.arange(64, dtype=np.float32)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    import base64

    payload = {
        "observation": {
            "images": {
                "front": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                }
            },
            "state": state.tolist(),
        },
        "task": "open the door and walk through it",
    }
    observation = bridge.build_gr00t_observation(payload)
    np.testing.assert_array_equal(observation["state"]["root_rot6d"][0, 0], state[0:6])
    np.testing.assert_array_equal(observation["state"]["joint_pos"][0, 0], state[6:35])
    np.testing.assert_array_equal(observation["state"]["joint_vel"][0, 0], state[35:64])
    assert observation["video"]["front"].shape == (1, 1, 8, 12, 3)
    assert observation["language"][bridge.LANGUAGE_KEY] == [["Open the door."]]


def test_http_bridge_matches_humanoidarena_task_language_routing():
    bridge = load_module("humanoidarena_http_bridge_language", "serve_http_bridge.py")
    assert bridge.canonical_task_instruction("Isaac-Move-Open-Door-G129-Dex3-Wholebody") == (
        "Open the door."
    )
    assert bridge.canonical_task_instruction("HSI_open_door") == "Open the door."


def test_dataset_validation_requires_sonic_64d_40d(tmp_path: Path):
    prepare = load_module("humanoidarena_prepare_dataset", "prepare_dataset.py")
    meta = tmp_path / "meta"
    meta.mkdir()
    info = {
        "codebase_version": "v2.1",
        "robot_type": prepare.ROBOT_TYPE,
        "features": {
            "observation.state": {"shape": [64]},
            "action": {"shape": [40]},
            "observation.images.front": {"shape": [480, 640, 3], "dtype": "video"},
        },
        "vla_protocol": {
            "schema": "unitree_g1_gmt_refpose_v3_1",
            "backend_source": "sonic",
            "action_semantics": "reference_pose_not_robot_current_residual",
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "modality.json").write_text(
        (EXAMPLE_DIR / "modality.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert prepare.validate_dataset(tmp_path) == info

    info["vla_protocol"]["backend_source"] = "twist2"
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="SONIC"):
        prepare.validate_dataset(tmp_path)


def test_training_config_uses_requested_gr00t_action_horizon():
    config = load_module("humanoidarena_config", "humanoidarena_config.py")
    assert config.humanoidarena_config["action"].delta_indices == list(range(30))

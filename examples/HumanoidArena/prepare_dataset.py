"""Prepare a SONIC-only HumanoidArena dataset for GR00T N1.7.

This copies a LeRobot V3 dataset, converts the copy to LeRobot V2.1 when needed,
adds GR00T modality metadata, and validates the canonical 64D/40D protocol.
The input dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


STATE_DIM = 64
ACTION_DIM = 40
ROBOT_TYPE = "unitree_g1_refpose_v3_1"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def feature_dim(info: dict, key: str) -> int:
    try:
        shape = info["features"][key]["shape"]
    except KeyError as exc:
        raise ValueError(f"Dataset is missing required feature {key!r}") from exc
    if len(shape) != 1:
        raise ValueError(f"Feature {key!r} must be one-dimensional, got shape={shape}")
    return int(shape[0])


def validate_dataset(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    modality_path = root / "meta" / "modality.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    if not modality_path.is_file():
        raise FileNotFoundError(modality_path)

    info = read_json(info_path)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"GR00T requires LeRobot v2.1, got {info.get('codebase_version')!r}")
    if feature_dim(info, "observation.state") != STATE_DIM:
        raise ValueError("HumanoidArena observation.state must be exactly 64D")
    if feature_dim(info, "action") != ACTION_DIM:
        raise ValueError("HumanoidArena action must be exactly 40D")
    if "observation.images.front" not in info.get("features", {}):
        raise ValueError("Dataset is missing observation.images.front")
    protocol = info.get("vla_protocol", {})
    if protocol.get("schema") != "unitree_g1_gmt_refpose_v3_1":
        raise ValueError("Dataset is not HumanoidArena unitree_g1_gmt_refpose_v3_1")
    if protocol.get("backend_source") != "sonic":
        raise ValueError("Only the SONIC subset is allowed for this GR00T benchmark")
    if protocol.get("action_semantics") != "reference_pose_not_robot_current_residual":
        raise ValueError("Unexpected action semantics; expected canonical reference-pose targets")

    modality = read_json(modality_path)
    expected = read_json(Path(__file__).with_name("modality.json"))
    if modality != expected:
        raise ValueError("meta/modality.json does not match the HumanoidArena V3.1 protocol")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, required=True, help="SONIC-only HumanoidArena LeRobot dataset"
    )
    parser.add_argument("--output", type=Path, required=True, help="GR00T-owned output dataset")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)

    info = read_json(output / "meta" / "info.json")
    version = info.get("codebase_version")
    if version == "v3.0":
        converter = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "lerobot_conversion"
            / "convert_v3_to_v2.py"
        )
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--repo-id",
                output.name,
                "--root",
                str(output.parent),
            ],
            check=True,
        )
        converted_backup = output.parent / f"{output.name}_v3.0"
        if converted_backup.is_dir():
            shutil.rmtree(converted_backup)
    elif version != "v2.1":
        raise ValueError(f"Unsupported LeRobot codebase_version: {version!r}")

    shutil.copy2(Path(__file__).with_name("modality.json"), output / "meta" / "modality.json")
    validated = validate_dataset(output)
    print(
        f"Prepared {output}: episodes={validated.get('total_episodes')} "
        f"frames={validated.get('total_frames')} state={STATE_DIM}D action={ACTION_DIM}D"
    )


if __name__ == "__main__":
    main()

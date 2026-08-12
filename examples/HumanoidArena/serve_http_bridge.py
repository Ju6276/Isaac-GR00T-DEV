"""Expose a GR00T N1.7 PolicyServer through HumanoidArena's HTTP protocol."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from gr00t.policy.server_client import PolicyClient
import numpy as np


STATE_SLICES = {
    "root_rot6d": slice(0, 6),
    "joint_pos": slice(6, 35),
    "joint_vel": slice(35, 64),
}
ACTION_KEYS = ["root_xy_delta", "root_z", "root_rot6d", "joint_pos", "hand_binary"]
ACTION_DIMS = [2, 1, 6, 29, 2]
ACTION_HORIZON = 30
LANGUAGE_KEY = "annotation.human.task_description"
TASK_LANGUAGE_INSTRUCTIONS = {
    "HOI_double_desk": "Put the hammer from the right table into the basket on the left table.",
    "HOI_football": "Kick the soccer ball into the goal.",
    "HOI_pp_box": "Move the box from the table onto the shelf.",
    "HSI_vision_navi": "Avoid obstacles and move to the yellow marked area.",
    "HSI_open_door": "Open the door.",
    "HSI_sit_sofa": "Sit on the sofa.",
    "HOI_grap_cup": "Place the orange drink can on the table into the basket on the tea table.",
    "HSI_boxing": "Strike the green markers on the punching bag.",
}
TASK_ALIASES = {
    "doubledesk": "HOI_double_desk",
    "football": "HOI_football",
    "ppbox": "HOI_pp_box",
    "visionnavi": "HSI_vision_navi",
    "opendoor": "HSI_open_door",
    "openthedoor": "HSI_open_door",
    "sitsofa": "HSI_sit_sofa",
    "grapcup": "HOI_grap_cup",
    "grabcup": "HOI_grap_cup",
    "boxing": "HSI_boxing",
}


def canonical_task_instruction(value: Any) -> str:
    """Match the language routing used by HumanoidArena's LeRobot server."""
    raw = str(value or "").strip()
    normalized = "".join(character for character in raw.lower() if character.isalnum())
    for task_name, instruction in TASK_LANGUAGE_INSTRUCTIONS.items():
        if normalized in {
            "".join(character for character in task_name.lower() if character.isalnum()),
            "".join(character for character in instruction.lower() if character.isalnum()),
        }:
            return instruction
    for alias, task_name in TASK_ALIASES.items():
        if alias in normalized:
            return TASK_LANGUAGE_INSTRUCTIONS[task_name]
    return raw or "perform the humanoid task"


def decode_image(spec: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in spec["shape"])
    dtype = np.dtype(spec["dtype"])
    image = np.frombuffer(base64.b64decode(spec["data_b64"]), dtype=dtype).reshape(shape)
    if dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"front image must be uint8 HWC RGB, got dtype={dtype}, shape={shape}")
    return np.ascontiguousarray(image)


def build_gr00t_observation(payload: dict[str, Any]) -> dict[str, Any]:
    observation = payload["observation"]
    image = decode_image(observation["images"]["front"])
    state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    if state.shape != (64,):
        raise ValueError(f"HumanoidArena state must be 64D, got {state.shape}")
    task = canonical_task_instruction(payload.get("task", payload.get("task_name")))
    return {
        "video": {"front": image[None, None]},
        "state": {key: state[value][None, None] for key, value in STATE_SLICES.items()},
        "language": {LANGUAGE_KEY: [[task]]},
    }


def concatenate_action_chunk(actions: dict[str, Any]) -> np.ndarray:
    chunks = []
    horizon = None
    for key, expected_dim in zip(ACTION_KEYS, ACTION_DIMS, strict=True):
        if key not in actions:
            raise ValueError(f"GR00T response is missing action key {key!r}")
        value = np.asarray(actions[key], dtype=np.float32)
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != expected_dim:
            raise ValueError(
                f"Action {key!r} must have shape [1, horizon, {expected_dim}], got {value.shape}"
            )
        horizon = value.shape[1] if horizon is None else horizon
        if value.shape[1] != horizon:
            raise ValueError("GR00T action modalities have inconsistent horizons")
        chunks.append(value[0])
    result = np.concatenate(chunks, axis=-1)
    if result.shape != (ACTION_HORIZON, 40):
        raise ValueError(
            f"Combined HumanoidArena action must be [{ACTION_HORIZON}, 40], "
            f"got {result.shape}"
        )
    return result


def make_handler(client: PolicyClient):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict[str, Any]) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length)) if length else {}
                if self.path == "/infer":
                    actions, _ = client.get_action(build_gr00t_observation(payload))
                    chunk = concatenate_action_chunk(actions)
                    self._reply(200, {"action_chunk": chunk.tolist()})
                elif self.path == "/reset":
                    client.reset({"seed": payload.get("seed")})
                    self._reply(200, {"ok": True})
                else:
                    self._reply(404, {"error": f"unknown endpoint: {self.path}"})
            except Exception as exc:
                self._reply(500, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[humanoidarena-http] {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gr00t-host", default="127.0.0.1")
    parser.add_argument("--gr00t-port", type=int, default=5555)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()

    client = PolicyClient(
        host=args.gr00t_host,
        port=args.gr00t_port,
        timeout_ms=args.timeout_ms,
        strict=False,
    )
    if not client.ping():
        raise RuntimeError(f"Cannot reach GR00T server at {args.gr00t_host}:{args.gr00t_port}")
    server = ThreadingHTTPServer((args.http_host, args.http_port), make_handler(client))
    print(f"HumanoidArena HTTP bridge listening on http://{args.http_host}:{args.http_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

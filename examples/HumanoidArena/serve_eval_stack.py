"""Launch a GR00T policy server plus the HumanoidArena HTTP bridge.

This accepts the server CLI expected by HumanoidArena's evaluation harness.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]


def free_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    backend_port = free_port("127.0.0.1")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    backend = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(REPO_ROOT / "gr00t/eval/run_gr00t_server.py"),
            "--model-path",
            str(Path(args.policy_path).resolve()),
            "--embodiment-tag",
            "HUMANOIDARENA_G1",
            "--device",
            args.device,
            "--host",
            "127.0.0.1",
            "--port",
            str(backend_port),
            "--no-strict",
        ],
        cwd=REPO_ROOT,
        env=environment,
    )
    bridge: subprocess.Popen | None = None

    def stop_children(*_: object) -> None:
        for process in (bridge, backend):
            if process is not None and process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    try:
        # The bridge performs a backend ping before opening its HTTP port.
        bridge = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(REPO_ROOT / "examples/HumanoidArena/serve_http_bridge.py"),
                "--gr00t-host",
                "127.0.0.1",
                "--gr00t-port",
                str(backend_port),
                "--http-host",
                args.host,
                "--http-port",
                str(args.port),
                "--timeout-ms",
                "360000",
            ],
            cwd=REPO_ROOT,
            env=environment,
        )
        while True:
            if backend.poll() is not None:
                raise RuntimeError(f"GR00T backend exited with status {backend.returncode}")
            if bridge.poll() is not None:
                raise RuntimeError(f"HTTP bridge exited with status {bridge.returncode}")
            time.sleep(1)
    finally:
        stop_children()
        for process in (bridge, backend):
            if process is not None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()

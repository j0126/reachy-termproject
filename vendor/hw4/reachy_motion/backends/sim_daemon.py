"""Start / stop a local MuJoCo simulation daemon and refuse to talk to anything else.

Safety interlocks (the real robot must never receive a motion command from this module):
  * The daemon is spawned by us from this venv with ``--sim`` and bound to 127.0.0.1
    (``--fastapi-host``; the daemon default for non-wireless is also loopback,
    reachy_mini/daemon/app/main.py:106-116).
  * A distinct robot name (``hw4_sim``) so its mDNS record (registered for every daemon,
    reachy_mini/utils/discovery.py:62-135) cannot be mistaken for the real robot ("reachy_mini").
  * ``assert_local_sim`` checks GET /api/daemon/status (reachy_mini/io/protocol.py:164-184):
    simulation_enabled is True, robot_name == hw4_sim, version == pinned SDK version.
    Callers must call it before connecting a ReachyMini client.
  * ``assert_motion_client_is_local_sim`` repeats that check on the WebSocket that will carry the
    motion. For a non-default robot_name, ``ReachyMini(connection_mode="localhost_only")`` connects
    to the literal host "localhost" and ignores ``host`` (reachy_mini/reachy_mini.py:471-476); on
    this PC "localhost" resolves to ::1 before 127.0.0.1, and the SDK verifies only robot_name on that
    socket (reachy_mini.py:596-606). ``start`` also refuses when anything listens on 127.0.0.1 or
    ::1 at the port.
Flags used: --sim [--headless] --no-media --no-wake-up-on-start --no-goto-sleep-on-stop
--no-preload-datasets --dataset-update-interval 0 (main.py:654-830). ``--headless`` is a store_true flag
that defaults to False (main.py:84, 715-720): without it the daemon opens the MuJoCo viewer window
(``mujoco.viewer.launch_passive``, daemon/backend/mujoco/backend.py:162-183). ``SimDaemon(headless=False)``
is for watching a motion; every interlock above is the same in both modes. The MuJoCo backend starts in
the sleep pose (observed 2026-11-10: pitch 0.41 rad, antennas +-3.05 rad).
"""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from reachy_motion import SDK_VERSION

SIM_HOST = "127.0.0.1"
LOOPBACK_HOSTS = ("127.0.0.1", "::1")
CLIENT_HOSTS_OK = ("localhost",) + LOOPBACK_HOSTS
SIM_ROBOT_NAME = "hw4_sim"
DEFAULT_PORT = 8000


def daemon_executable() -> Path:
    """reachy-mini-daemon from the *current* interpreter's environment (this venv)."""
    exe = Path(sys.executable).with_name("reachy-mini-daemon.exe" if sys.platform == "win32" else "reachy-mini-daemon")
    if not exe.exists():
        raise FileNotFoundError(f"reachy-mini-daemon not found next to {sys.executable}")
    return exe


def _get_json(path: str, port: int, timeout: float) -> dict | None:
    """Read-only GET http://127.0.0.1:<port><path>; None if nothing answers or the answer is not JSON."""
    url = f"http://{SIM_HOST}:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, http.client.HTTPException):
        return None  # ValueError covers a non-JSON answer (json.JSONDecodeError)


def get_status(port: int = DEFAULT_PORT, timeout: float = 2.0) -> dict | None:
    """GET http://127.0.0.1:<port>/api/daemon/status (read-only); None if nothing answers."""
    return _get_json("/api/daemon/status", port, timeout)


def get_kinematics_info(port: int = DEFAULT_PORT, timeout: float = 2.0) -> dict | None:
    """GET http://127.0.0.1:<port>/api/kinematics/info (read-only): IK engine + collision-check flag.

    The daemon answers ``{"info": {"engine": ..., "collision check": ...}}``
    (reachy_mini/daemon/app/routers/kinematics.py:29-39, mounted under /api in main.py:312-318).
    """
    return _get_json("/api/kinematics/info", port, timeout)


def port_answers(host: str, port: int, timeout: float = 0.5) -> bool:
    """True if anything accepts a TCP connection on host:port (sends no data)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:  # refused, timed out, or no IPv6 on this machine
        return False


def assert_local_sim(status: dict | None, expected_version: str = SDK_VERSION) -> dict:
    """Raise unless `status` describes OUR local MuJoCo sim daemon."""
    if status is None:
        raise RuntimeError("no daemon answered on 127.0.0.1")
    problems = []
    if status.get("simulation_enabled") is not True:
        problems.append(f"simulation_enabled={status.get('simulation_enabled')!r}")
    if status.get("robot_name") != SIM_ROBOT_NAME:
        problems.append(f"robot_name={status.get('robot_name')!r} (expected {SIM_ROBOT_NAME!r})")
    if status.get("version") != expected_version:
        problems.append(f"version={status.get('version')!r} (expected {expected_version!r})")
    if problems:
        raise RuntimeError("refusing to send motion: not the local sim daemon: " + ", ".join(problems))
    return status


def assert_motion_client_is_local_sim(mini: object, expected_version: str = SDK_VERSION,
                                      timeout: float = 5.0) -> dict:
    """Run the interlock on the daemon behind the client's own WebSocket (call before any motion).

    The daemon pushes DaemonStatus to every /ws/sdk client at 1 Hz (reachy_mini/io/ws_server.py:105-110,
    daemon/daemon.py:829-840); ``WSClient.get_status`` waits for the next one
    (io/ws_client.py:228-234).
    """
    client = getattr(mini, "client", None)
    host = getattr(client, "host", None)
    if host not in CLIENT_HOSTS_OK:
        raise RuntimeError(f"refusing to send motion: client host {host!r} is not loopback")
    status = client.get_status(timeout=timeout)
    data = status.model_dump(mode="json") if hasattr(status, "model_dump") else dict(status)
    return assert_local_sim(data, expected_version)


class SimDaemon:
    """Context manager that owns one MuJoCo daemon process on 127.0.0.1 (headless unless asked otherwise)."""

    def __init__(self, port: int = DEFAULT_PORT, log_path: Path | None = None, scene: str = "empty",
                 headless: bool = True):
        self.port = port
        self.log_path = log_path
        self.scene = scene
        self.headless = bool(headless)  # False: the daemon opens the MuJoCo viewer window (backend.py:170-183)
        self.proc: subprocess.Popen | None = None
        self._log_file = None

    def command(self, platform: str | None = None) -> list[str]:
        """The daemon command line. macOS: the viewer (``mujoco.viewer.launch_passive``, mujoco/backend.py:171) runs
        only under ``mjpython`` (MuJoCo's rule for macOS), so with the viewer the daemon module is started by the
        ``mjpython`` of this venv; without one the daemon runs headless. Windows / Linux: the console script."""
        platform = sys.platform if platform is None else platform
        head, headless = [str(daemon_executable())], self.headless
        if platform == "darwin" and not headless:
            mjpython = Path(sys.executable).with_name("mjpython")
            if mjpython.exists():
                head = [str(mjpython), "-m", "reachy_mini.daemon.app.main"]  # main.py:871 runs main()
            else:
                headless = True
        return [
            *head,
            "--sim", *(["--headless"] if headless else []), "--no-media",
            "--no-wake-up-on-start", "--no-goto-sleep-on-stop",
            "--no-preload-datasets", "--dataset-update-interval", "0",
            "--scene", self.scene,
            "--robot-name", SIM_ROBOT_NAME,
            "--fastapi-host", SIM_HOST, "--fastapi-port", str(self.port),
            "--log-level", "INFO",
        ]

    def start(self, timeout: float = 60.0) -> dict:
        busy = [h for h in LOOPBACK_HOSTS if port_answers(h, self.port)]
        if busy:
            raise RuntimeError(f"something already listens on port {self.port} at {busy}; stop it first")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        try:
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file = self.log_path.open("w", encoding="utf-8")
            self.proc = subprocess.Popen(
                self.command(),
                stdout=self._log_file or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            deadline = time.perf_counter() + timeout
            while time.perf_counter() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"sim daemon exited early with code {self.proc.returncode}")
                status = get_status(self.port)
                if status is not None and status.get("state") == "running":
                    return assert_local_sim(status)
                time.sleep(0.25)
            raise TimeoutError("sim daemon did not become ready")
        except BaseException:  # interlock refusal, Ctrl+C, timeout, ...: never orphan the daemon
            self.stop()
            raise

    def stop(self, timeout: float = 10.0) -> None:
        try:
            if self.proc is not None and self.proc.poll() is None:
                if sys.platform == "win32":
                    import signal

                    try:
                        self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # graceful (lifespan cleanup)
                        self.proc.wait(timeout=timeout)
                    except (subprocess.TimeoutExpired, OSError, ValueError):
                        self.proc.kill()
                        self.proc.wait(timeout=timeout)
                else:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=timeout)
        finally:
            self.proc = None
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def __enter__(self) -> "SimDaemon":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

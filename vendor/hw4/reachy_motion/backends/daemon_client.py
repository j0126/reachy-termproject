"""Read-only REST client for a Reachy Mini daemon: GET only, on a fixed whitelist of paths plus one exact query.

Adapted from HW3 ``../reachy_hw3/reachy_s2s/robot/connection.py`` ``DaemonClient`` (HW3 HEAD 0d49b6b): ``requests``
replaced by the stdlib ``urllib`` (no extra dependency), ``post`` / ``set_motor_mode`` and the volume / audio / DoA
helpers removed, ``host`` made mandatory (it never defaults to the robot), errors wrapped in ``BackendError``.

CLAUDE.md rule 3 allows exactly these REST GETs on the robot daemon without approval, so they are the only requests
this class can make (``ALLOWED_GET_PATHS``; SDK 1.11.0 routes under the "/api" prefix, daemon/app/main.py:312-333;
SDK paths here are relative to ``site-packages/reachy_mini/``, ``routers/`` = daemon/app/routers/):

* ``/api/daemon/status``          DaemonStatus (daemon/app/routers/daemon.py:84-87; io/protocol.py:164-184)
* ``/api/kinematics/info``        IK engine and collision flag (routers/kinematics.py:29-39)
* ``/api/state/full``             control mode, head pose (m, rad), body yaw, antennas (routers/state.py:88-142)
* ``/api/state/present_body_yaw`` rad (routers/state.py:43-48)
* ``/api/motors/status``          {"mode": "enabled" | "disabled" | "gravity_compensation"} (routers/motors.py:29-32;
  io/protocol.py:43-48)

``present_head_pose()`` reads the ``head_pose`` field of /api/state/full (defaults ``with_head_pose=True``,
``use_pose_matrix=False``: XYZRPYPose in m / rad with ``as_euler("xyz")``, daemon/app/models.py:68-93 - the Pose6
convention): /api/state/present_head_pose is not on the CLAUDE.md list.

One query string exists, sent only exactly as written (``ALLOWED_GET_QUERIES``; added to the CLAUDE.md list by the
user on 2026-11-11 for the R0 start-pose check):

* ``/api/state/full?with_head_joints=true``  the defaults above plus ``head_joints`` = [body_yaw, stewart_1..6] in rad
  (routers/state.py:118-119; FullState.head_joints, daemon/app/models.py:154). On the robot it returns
  ``get_all_joint_positions()[0]`` = the motor controller's last read positions (``c.get_last_position()``,
  daemon/backend/robot/backend.py:480-506): a cached value, nothing is written and no bus transfer is started.

Every other query string is refused. The defaults of /api/state/full only read; ``with_doa=true`` registers demand
that (re)starts the daemon's DoA poller (daemon/backend/abstract.py:1553-1571), ``with_target_head_pose=true`` fails
with HTTP 500 when no target exists (routers/state.py:114-117), and the ``with_target_*`` fields are dropped from the
answer anyway (FullState, daemon/app/models.py:149-160).

There is deliberately no POST / PUT / DELETE method: motion, motor modes, daemon start / stop / restart and media
all change robot state and need an explicit approval (CLAUDE.md rule 3).

The request sent is exactly the one built in ``get()``: it goes through ``_OPENER``, urllib's default handlers with
two changes. Redirects are never followed: the default ``HTTPRedirectHandler`` would send a second GET to whatever
``Location`` the server names (any host, path or query string) before the answer could be checked; here a 3xx answer
is a ``BackendError`` and nothing else is requested (the daemon's own routes never redirect). No proxy is used:
urllib otherwise takes them from the environment and, on Windows, from the system settings
(``urllib.request.getproxies``), and the daemon is on localhost or the LAN. ``install_opener`` is never called, so
the process-wide ``urlopen`` is left as it is.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import numbers
import re
import urllib.error
import urllib.request
from typing import Any

from reachy_motion.backends.base import BackendError
from reachy_motion.library.pose import Pose6

STATUS_PATH = "/api/daemon/status"
KINEMATICS_INFO_PATH = "/api/kinematics/info"
STATE_FULL_PATH = "/api/state/full"
PRESENT_BODY_YAW_PATH = "/api/state/present_body_yaw"
MOTORS_STATUS_PATH = "/api/motors/status"

#: Every request this client can make: GET, exactly these paths without a query string, or exactly one of
#: ``ALLOWED_GET_QUERIES``; one request per call - redirects are not followed and no proxy is used (``_OPENER``;
#: CLAUDE.md rule 3).
ALLOWED_GET_PATHS: frozenset[str] = frozenset(
    {STATUS_PATH, KINEMATICS_INFO_PATH, STATE_FULL_PATH, PRESENT_BODY_YAW_PATH, MOTORS_STATUS_PATH})
HEAD_JOINTS_QUERY = STATE_FULL_PATH + "?with_head_joints=true"  # cached joint read (module docstring)
ALLOWED_GET_QUERIES: frozenset[str] = frozenset({HEAD_JOINTS_QUERY})

DEFAULT_PORT = 8000  # daemon FastAPI port (backends.sim_daemon.DEFAULT_PORT)

_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_HOSTNAME = re.compile(rf"(?=.{{1,253}}$){_LABEL}(?:\.{_LABEL})*\.?")
_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follows a redirect: returning None from ``redirect_request`` makes urllib hand the 3xx answer to its
    default error handler, which raises ``HTTPError`` - before any second request is built."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> None:
        return None


#: The only opener this module sends with (module docstring): the urllib defaults, the redirect handler replaced by
#: ``_RefuseRedirects`` and the proxy handler by one without proxies.
_OPENER = urllib.request.build_opener(_RefuseRedirects, urllib.request.ProxyHandler({}))


def _url_host(host: object) -> str:
    """``host`` as written in the URL; refuses anything that could smuggle a scheme, port, path, query or user."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host is required: an IP address or a host name (there is deliberately no default)")
    h = host.strip()
    bracketed = h.startswith("[") and h.endswith("]")
    try:
        ip = ipaddress.ip_address(h[1:-1] if bracketed else h)
    except ValueError:
        if bracketed or not _HOSTNAME.fullmatch(h):
            raise ValueError(f"invalid host {host!r}: give a bare IP address or host name "
                             "(no scheme, port, path, query or credentials)") from None
        return h
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.scope_id:
            raise ValueError(f"scoped IPv6 address {host!r} is not supported")
        return f"[{ip.compressed}]"
    if bracketed:
        raise ValueError(f"invalid host {host!r}: brackets are for IPv6 addresses only")
    return ip.compressed


def _timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not (
            math.isfinite(float(value)) and float(value) > 0):
        raise ValueError(f"timeout_s must be a positive finite number of seconds, got {value!r}")
    return float(value)


def loop_stats(status: Any) -> dict[str, Any] | None:
    """``backend_status.control_loop_stats`` of a DaemonStatus dict, or None when absent or empty.

    Only the robot backend reports it (RobotBackendStatus, io/protocol.py:67-74; filled once per second with
    mean_control_loop_frequency = mean(1/dt), max_control_loop_interval, nb_error, motor_controller,
    daemon/backend/robot/backend.py:305-319); the MuJoCo and mockup statuses have no such field
    (io/protocol.py:77-88).
    """
    backend = status.get("backend_status") if isinstance(status, dict) else None
    stats = backend.get("control_loop_stats") if isinstance(backend, dict) else None
    return dict(stats) if isinstance(stats, dict) and stats else None


class DaemonClient:
    """Read-only REST client: GET on ``ALLOWED_GET_PATHS`` only (module docstring). ``host`` is required."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout_s: float = 5.0) -> None:
        self.host = _url_host(host)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"port must be an integer in 1..65535, got {port!r}")
        self.port = port
        self.timeout_s = _timeout(timeout_s)
        self.base_url = f"http://{self.host}:{port}"

    def __repr__(self) -> str:
        return f"DaemonClient({self.base_url!r}, read-only GET)"

    # ------------------------------------------------------------------ the only request
    def get(self, path: str, timeout_s: float | None = None) -> Any:
        """GET ``path`` (one of ``ALLOWED_GET_PATHS``, or exactly one of ``ALLOWED_GET_QUERIES``) and return the
        decoded JSON answer."""
        if not isinstance(path, str):
            raise ValueError(f"path must be a string, got {type(path).__name__}")
        if path in ALLOWED_GET_QUERIES:
            pass  # exact string match only: no other parameter, value spelling or order
        elif "?" in path or "#" in path:
            raise ValueError(f"query strings are refused except exactly {sorted(ALLOWED_GET_QUERIES)}: {path!r}")
        elif path not in ALLOWED_GET_PATHS:
            raise ValueError(f"{path!r} is not one of the read-only paths {sorted(ALLOWED_GET_PATHS)} "
                             "(CLAUDE.md rule 3)")
        url = self.base_url + path
        timeout = self.timeout_s if timeout_s is None else _timeout(timeout_s)
        request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with _OPENER.open(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:  # a redirect: not followed (_RefuseRedirects), nothing else was requested
                location = exc.headers.get("Location") if exc.headers is not None else None
                raise BackendError(f"GET {url} answered HTTP {exc.code} (redirect to {location!r}): not followed, "
                                   "refused (read-only client)") from exc
            raise BackendError(f"GET {url} failed: HTTP {exc.code} {exc.reason}") from exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:  # OSError: timeouts, resets
            raise BackendError(f"GET {url} failed: {exc}") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
            raise BackendError(f"GET {url}: the answer is not JSON ({exc})") from exc

    def _get_object(self, path: str, timeout_s: float | None) -> dict[str, Any]:
        value = self.get(path, timeout_s)
        if not isinstance(value, dict):
            raise BackendError(f"GET {self.base_url}{path}: expected a JSON object, got {type(value).__name__}")
        return value

    # ------------------------------------------------------------------ readers
    def status(self, timeout_s: float | None = None) -> dict[str, Any]:
        """DaemonStatus: robot_name, state, version, simulation_enabled, mockup_sim_enabled, backend_status, ..."""
        return self._get_object(STATUS_PATH, timeout_s)

    def kinematics_info(self, timeout_s: float | None = None) -> dict[str, Any]:
        """{"info": {"engine": ..., "collision check": ...}}."""
        return self._get_object(KINEMATICS_INFO_PATH, timeout_s)

    def state_full(self, timeout_s: float | None = None) -> dict[str, Any]:
        """FullState with the default fields: control_mode, head_pose, body_yaw, antennas_position, timestamp (UTC)."""
        return self._get_object(STATE_FULL_PATH, timeout_s)

    def state_full_with_head_joints(self, timeout_s: float | None = None) -> dict[str, Any]:
        """FullState defaults plus ``head_joints`` ([body_yaw, stewart_1..6], rad): pose and joints of one snapshot."""
        return self._get_object(HEAD_JOINTS_QUERY, timeout_s)

    def present_head_joints(self, timeout_s: float | None = None) -> list[float]:
        """The 7 present head joints [body_yaw, stewart_1..6] (rad), cached by the daemon (module docstring)."""
        joints = self.state_full_with_head_joints(timeout_s).get("head_joints")
        if (not isinstance(joints, list) or len(joints) != 7
                or not all(isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(q) for q in joints)):
            raise BackendError(f"GET {self.base_url}{HEAD_JOINTS_QUERY}: expected 7 finite head_joints, got {joints!r}")
        return [float(q) for q in joints]

    def present_head_pose(self, timeout_s: float | None = None) -> Pose6:
        """Present head pose (m, rad) from the ``head_pose`` field of /api/state/full."""
        pose = self.state_full(timeout_s).get("head_pose")
        try:
            return Pose6(*(float(pose[k]) for k in _POSE_KEYS))  # type: ignore[index]
        except (TypeError, KeyError, ValueError) as exc:
            raise BackendError(f"GET {self.base_url}{STATE_FULL_PATH}: no x/y/z/roll/pitch/yaw head_pose in the "
                               f"answer ({pose!r})") from exc

    def present_body_yaw(self, timeout_s: float | None = None) -> float:
        """Present body yaw (rad)."""
        value = self.get(PRESENT_BODY_YAW_PATH, timeout_s)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BackendError(f"GET {self.base_url}{PRESENT_BODY_YAW_PATH}: expected a number, got {value!r}")
        return float(value)

    def motors_status(self, timeout_s: float | None = None) -> dict[str, Any]:
        """{"mode": "enabled" | "disabled" | "gravity_compensation"}."""
        return self._get_object(MOTORS_STATUS_PATH, timeout_s)

    def motor_mode(self, timeout_s: float | None = None) -> str | None:
        """The motor control mode string, or None if the answer has none."""
        mode = self.motors_status(timeout_s).get("mode")
        return mode if isinstance(mode, str) else None

    def control_loop_stats(self, timeout_s: float | None = None) -> dict[str, Any] | None:
        """The robot backend's control-loop statistics (None for MuJoCo / mockup or before the first second)."""
        return loop_stats(self.status(timeout_s))

    def summary(self, timeout_s: float | None = None) -> dict[str, Any]:
        """Daemon facts worth storing with an experiment, best effort: failures become ``*_error`` strings.

        ``data_kind`` is "simulated" when the daemon reports ``simulation_enabled``, "measured" when it reports
        neither simulation nor mockup, None otherwise (CLAUDE.md rule 6 labels).
        """
        info: dict[str, Any] = {"daemon_url": self.base_url, "client": "read-only GET"}
        try:
            st = self.status(timeout_s)
        except BackendError as exc:
            info["status_error"] = str(exc)
            return info
        backend = st.get("backend_status")
        sim, mockup = st.get("simulation_enabled"), st.get("mockup_sim_enabled")
        info.update(
            robot_name=st.get("robot_name"),
            daemon_version=st.get("version"),
            state=st.get("state"),
            simulation_enabled=sim,
            mockup_sim_enabled=mockup,
            wireless=st.get("wireless_version"),
            no_media=st.get("no_media"),
            wlan_ip=st.get("wlan_ip"),
            error=st.get("error"),
            motor_control_mode=backend.get("motor_control_mode") if isinstance(backend, dict) else None,
            control_loop_stats=loop_stats(st),
            data_kind="simulated" if sim is True else "measured" if (sim is False and mockup is False) else None,
        )
        try:
            info["kinematics"] = self.kinematics_info(timeout_s).get("info")
        except BackendError as exc:
            info["kinematics_error"] = str(exc)
        return info

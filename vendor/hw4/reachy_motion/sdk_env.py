"""Environment facts: installed package versions and the pins this project relies on.

Used by ``reachy-motion env`` and by tests/test_sdk_env.py so that an accidental SDK
upgrade (which would invalidate the Phase 1 source references) is caught immediately.
Nothing here touches the network or the robot.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from reachy_motion import SDK_VERSION

# Exact versions the Phase 1 analysis was done against (verified 2026-11-10 in .venv).
EXPECTED_PINS: dict[str, str] = {
    "reachy-mini": SDK_VERSION,
    "reachy-mini-rust-kinematics": "1.0.3",
    "reachy-mini-motor-controller": "1.5.6",
}

# Recorded for reproducibility only (not enforced).
REPORTED_PACKAGES: tuple[str, ...] = (
    "reachy-mini",
    "reachy-mini-rust-kinematics",
    "reachy-mini-motor-controller",
    "rustypot",
    "numpy",
    "mujoco",
    "matplotlib",
)


def installed_version(dist: str) -> str | None:
    """Installed version of a distribution, or None if it is not installed."""
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def environment_report() -> dict[str, object]:
    """Versions + interpreter info, suitable for a results manifest."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {d: installed_version(d) for d in REPORTED_PACKAGES},
    }


def pin_mismatches() -> dict[str, tuple[str, str | None]]:
    """{dist: (expected, installed)} for every pin that does not match."""
    out: dict[str, tuple[str, str | None]] = {}
    for dist, expected in EXPECTED_PINS.items():
        got = installed_version(dist)
        if got != expected:
            out[dist] = (expected, got)
    return out

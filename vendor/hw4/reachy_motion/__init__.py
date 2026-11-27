"""reachy_motion - expressive motion toolkit for Reachy Mini (DGIST HRI HW4).

Layers (see docs/ARCHITECTURE.md):
    library    - motion specs (keyframes, timing, easing, V/A metadata)      [Phase 2]
    executor   - plays a spec through a backend, logs what was commanded     [Phase 2]
    safety     - limits / IK feasibility / speed checks before execution     [Phase 2-3]
    backends   - mock (no daemon), MuJoCo sim daemon, real robot (armed)     [Phase 2]
    telemetry  - commanded vs measured logging, RMSE / peak error / lag      [Phase 3]
    analysis   - SDK analysis reproductions and report figures               [Phase 1]

Phase 1 provides ``analysis``, ``backends.sim_daemon`` (local MuJoCo daemon + interlock), ``cli`` and
``sdk_env``. Nothing in this package drives the real robot; motion is sent only to a local sim daemon
that reports simulation_enabled=True.
"""

__version__ = "0.1.0"

# The SDK version every source reference in docs/phase1 was taken from.
SDK_VERSION = "1.11.0"

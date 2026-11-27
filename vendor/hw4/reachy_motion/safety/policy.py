"""Initial safety POLICY (Phase 2): thresholds a plan must meet before any backend may run it.

These numbers are an initial, conservative engineering policy - NOT a proof of mechanical safety:

* Joint limits come from the SDK URDF (= MJCF = EEPROM ticks through an INFERRED raw-to-angle mapping,
  Phase 1 SDK_ANALYSIS 6.1, 8). The 10 deg margin is a buffer chosen by us, not a measured clearance.
* There is no collision model (the daemon's collision check is off, the MJCF head-body contacts are
  only in MuJoCo); the Stewart rods, head shell and body can still interfere inside the limits.
* Motor speeds are compared with ROBOTIS NO-LOAD datasheet speeds at 5 V (an upper bound - speed
  drops with load; the robot's bus voltage is undocumented; the body motor's datasheet is a proxy).
  Half of an upper bound is not a guaranteed trackable speed.
* Accelerations, torques, temperatures and the motors' own Profile settings are not modelled.

A plan that passes is "within the initial policy". Whether it is safe on the robot is decided only
by later staged tests with explicit approval (Phase 4, CLAUDE.md rule 3), small amplitude first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

POLICY_DISCLAIMER = (
    "Initial Phase 2 policy, not a guarantee of mechanical safety: URDF limits via an inferred raw-angle "
    "mapping, no collision model, no-load datasheet speeds at 5 V (upper bound, unknown bus voltage), no "
    "acceleration/torque model. Passing means 'within policy', not 'safe on the robot'."
)


@dataclass(frozen=True)
class SafetyPolicy:
    name: str = "phase2-initial-v1"
    # --- joint space (IK of every checked sample, SDK AnalyticalKinematics with automatic body yaw)
    min_stewart_margin_deg: float = 10.0  # distance to the URDF limit of every Stewart joint
    max_stewart_speed_ratio: float = 0.5  # x XL330-M288-T no-load speed at 5 V (618 deg/s)
    max_body_speed_ratio: float = 0.5  # x XC330-M288-T (proxy) no-load speed at 5 V (486 deg/s)
    max_antenna_speed_ratio: float = 0.5  # x XL330-M077-T no-load speed at 5 V (2298 deg/s)
    max_auto_body_shift_deg: float = 2.0  # |IK body joint - commanded body yaw|: automatic body yaw engaging
    max_body_step_deg: float = 5.0  # per-sample body joint change at the check rate (catches the psi wrap flip)
    max_psi_step_deg: float = 90.0  # per-sample change of psi = atan2(-R01, R00) (wrap detector, Phase 1 R1)
    # --- task space (official DOF table + Phase 1 IK feasibility; SDK_ANALYSIS 6.4)
    max_roll_deg: float = 40.0
    max_pitch_deg: float = 40.0
    max_rel_yaw_deg: float = 60.0  # head yaw relative to the body (the IK clamps at 65 deg)
    max_body_yaw_deg: float = 155.0
    x_range_mm: tuple[float, float] = (-15.0, 25.0)
    y_range_mm: tuple[float, float] = (-40.0, 40.0)
    z_range_mm: tuple[float, float] = (-40.0, 24.0)  # IK becomes NaN from z = +24.3 mm (Phase 1 6.3)
    max_antenna_abs_rad: float = 3.05  # SDK sleep pose angle; URDF allows +-pi
    # --- sampling
    check_rate_hz: float = 200.0  # dense samples for peaks (cartoon overshoot, speed peaks)
    stream_rate_hz: float = 50.0  # the executor's tick rate (per-tick steps are checked on this grid too)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY = SafetyPolicy()


@dataclass(frozen=True)
class Violation:
    code: str  # e.g. "ik_nan", "stewart_margin", "stewart_speed", "envelope_pitch", "psi_wrap"
    message: str
    t: float | None = None  # s within the plan (first occurrence)
    value: float | None = None  # worst value found
    limit: float | None = None  # policy threshold it was compared with
    severity: str = "violation"  # "violation" (plan refused) | "warning" (reported only)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyReport:
    name: str
    variant: str
    policy: SafetyPolicy
    passes_policy: bool
    violations: tuple[Violation, ...]
    warnings: tuple[Violation, ...]
    metrics: dict[str, Any]  # worst values with units in the key names (e.g. "min_stewart_margin_deg")
    spec_digest: str = ""
    timeline_digest: str = ""
    data_kind: str = "computed"
    disclaimer: str = POLICY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "variant": self.variant, "passes_policy": self.passes_policy,
            "policy": self.policy.to_dict(), "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings], "metrics": self.metrics,
            "spec_digest": self.spec_digest, "timeline_digest": self.timeline_digest,
            "data_kind": self.data_kind, "disclaimer": self.disclaimer,
        }

"""Joint limits and motor specs used by the Phase 1 hardware-constraint analysis.

Joint limits are read from the SDK's own URDF at run time (not hard-coded):
    reachy_mini/descriptions/reachy_mini/urdf/robot.urdf  (stewart_1..6, yaw_body, antennas)
The same ranges appear in the MJCF and, as raw ticks, in assets/config/hardware_config.yaml
(EEPROM limits); see docs/phase1/SDK_ANALYSIS.md section 6.1. URDF velocity/effort values are
generator placeholders and are NOT used.

Motor specs are datasheet values (ROBOTIS e-manual) for the models named by the official
v1.11.0 hardware page; they are not in the SDK. The robot runs the motors above the rated
6 V on purpose and the real bus voltage is undocumented, so the 5 V / 6 V no-load speeds are
only reference points (an upper bound: speed drops with load torque).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import reachy_mini

from reachy_motion.analysis.goto_replay import JOINT_NAMES

URDF_JOINT_NAMES: dict[str, str] = {name: name for name in JOINT_NAMES[1:]} | {"body_yaw": "yaw_body"}


@lru_cache(maxsize=1)
def urdf_joint_limits() -> dict[str, tuple[float, float]]:
    """{joint (our naming): (lower_rad, upper_rad)} for body_yaw and stewart_1..6 from the SDK URDF."""
    path = files(reachy_mini).joinpath("descriptions/reachy_mini/urdf/robot.urdf")
    root = ET.parse(str(path)).getroot()
    by_name = {j.get("name"): j for j in root.iter("joint")}
    out: dict[str, tuple[float, float]] = {}
    for ours in JOINT_NAMES:
        joint = by_name[URDF_JOINT_NAMES[ours]]
        lim = joint.find("limit")
        out[ours] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


@dataclass(frozen=True)
class MotorSpec:
    model: str
    no_load_rpm_5v: float
    no_load_rpm_6v: float
    stall_torque_nm_5v: float
    stall_torque_nm_6v: float
    source: str

    @property
    def no_load_deg_s_5v(self) -> float:
        return self.no_load_rpm_5v * 6.0

    @property
    def no_load_deg_s_6v(self) -> float:
        return self.no_load_rpm_6v * 6.0


# Official v1.11.0 hardware page: Stewart 6x XL330-M288-T, antennas 2x XL330-M077-T,
# body 1x custom XC330-M288-PG (plastic gear; the metal-gear XC330-M288-T datasheet is a proxy).
STEWART_MOTOR = MotorSpec("XL330-M288-T", 103, 123, 0.52, 0.60,
                          "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/")
ANTENNA_MOTOR = MotorSpec("XL330-M077-T", 383, 456, 0.215, 0.228,
                          "https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/")
BODY_MOTOR_PROXY = MotorSpec("XC330-M288-T (proxy for custom XC330-M288-PG)", 81, 97, 0.93, 1.10,
                             "https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/")

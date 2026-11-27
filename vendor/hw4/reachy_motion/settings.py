"""Project settings: the one place that names this project's robot.

The user confirmed the robot's address on 2026-11-11. Naming the robot here does NOT allow anything: every motor
action still needs ``--armed``, an approval file the user wrote and a typed confirmation (CLAUDE.md rule 3,
``safety.approval``), and the read-only R0 check (``robot-status``) only sends the whitelisted GETs
(``backends.daemon_client``).
"""

from __future__ import annotations

ROBOT_HOST = "192.168.0.15"  # Reachy Mini Wireless daemon, confirmed by the user 2026-11-11
ROBOT_HOSTNAME = "reachy-mini.local"  # its mDNS name (documentation only: commands use ROBOT_HOST)
ROBOT_DAEMON_PORT = 8000  # daemon FastAPI port (backends.daemon_client.DEFAULT_PORT)

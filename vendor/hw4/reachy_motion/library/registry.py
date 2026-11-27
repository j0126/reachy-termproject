"""Motion library: the registry of ``MotionSpec`` by (name, variant)."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from reachy_motion.library.spec import MOTION_NAMES, VARIANTS, MotionSpec

#: Environment variable that points the library at another folder of motion files.
MOTION_DIR_ENV = "REACHY_MOTION_DIR"


def motions_dir() -> Path:
    """Folder of the editable motion files: ``$REACHY_MOTION_DIR`` or ``<repo>/motions``.

    The repo folder is found next to the package (the project is installed editable, CLAUDE.md).
    Files are named ``<name>.<variant>.yaml`` (degrees / millimetres, see ``library.io``).
    """
    env = os.environ.get(MOTION_DIR_ENV)
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "motions"


class MotionLibrary:
    def __init__(self, specs: Iterable[MotionSpec] = ()) -> None:
        self._specs: dict[tuple[str, str], MotionSpec] = {}
        for s in specs:
            self.register(s)

    def register(self, spec: MotionSpec) -> None:
        problems = spec.validate()
        if problems:
            raise ValueError(f"invalid spec {spec.key}: " + "; ".join(problems))
        k = (spec.name, spec.variant)
        if k in self._specs:
            raise ValueError(f"duplicate spec {spec.key}")
        self._specs[k] = spec

    def get(self, name: str, variant: str = "expressive") -> MotionSpec:
        try:
            return self._specs[(name, variant)]
        except KeyError:
            raise KeyError(f"no motion {name}:{variant} (have {sorted(self.keys())})") from None

    def keys(self) -> list[str]:
        return [f"{n}:{v}" for n, v in self._specs]

    def __iter__(self) -> Iterator[MotionSpec]:
        return iter(sorted(self._specs.values(), key=lambda s: (MOTION_NAMES.index(s.name) if s.name in MOTION_NAMES
                                                                 else 99, VARIANTS.index(s.variant)
                                                                 if s.variant in VARIANTS else 99)))

    def __len__(self) -> int:
        return len(self._specs)

    def missing(self) -> list[str]:
        """``name:variant`` pairs required by the HW4 assignment that are not registered."""
        return [f"{n}:{v}" for n in MOTION_NAMES for v in VARIANTS if (n, v) not in self._specs]


def default_library(directory: Path | None = None) -> MotionLibrary:
    """Every ``*.yaml`` motion file in ``directory`` (default ``motions_dir()``): A ``aha_recommend``,
    B ``boredom``, C ``success`` / ``failure`` x (expressive, neutral)."""
    from reachy_motion.library.io import load_spec_file  # local import: io needs PyYAML

    d = Path(directory) if directory is not None else motions_dir()
    if not d.is_dir():
        raise FileNotFoundError(f"motion folder not found: {d}")
    return MotionLibrary(load_spec_file(f) for f in sorted(d.glob("*.yaml")))

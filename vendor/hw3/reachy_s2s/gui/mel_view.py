"""Scrolling log-mel spectrogram widget (renders :class:`MelMonitor` snapshots)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

_TICK_HZ = (100, 250, 500, 1000, 2000, 4000, 8000)


class MelSpectrogramView(pg.PlotWidget):
    def __init__(
        self,
        n_mels: int,
        display_seconds: float,
        center_freqs_hz: np.ndarray,
        db_range: tuple[float, float],
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self._n_mels = n_mels
        self._levels = db_range
        self.auto_levels = False
        self._image = pg.ImageItem(axisOrder="col-major")  # image[x=time, y=mel]
        try:
            self._image.setColorMap(pg.colormap.get("inferno"))
        except Exception:  # colormap file missing in an unusual pyqtgraph build
            self._image.setColorMap(pg.colormap.get("viridis"))
        self.addItem(self._image)
        self._rect = QtCore.QRectF(-display_seconds, 0.0, display_seconds, float(n_mels))
        self._shape: tuple[int, ...] | None = None  # setRect needs an image first
        self.setXRange(-display_seconds, 0.0, padding=0)
        self.setYRange(0.0, float(n_mels), padding=0)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setLabel("bottom", "time (s)")
        self.setLabel("left", "frequency (Hz, mel scale)")
        positions = np.arange(n_mels) + 0.5
        ticks = [
            (float(np.interp(hz, center_freqs_hz, positions)), f"{hz // 1000}k" if hz >= 1000 else str(hz))
            for hz in _TICK_HZ
            if center_freqs_hz[0] <= hz <= center_freqs_hz[-1]
        ]
        self.getAxis("left").setTicks([ticks])
        self._colorbar = pg.ColorBarItem(values=db_range, colorMap=self._image.getColorMap(),
                                         label="dB", interactive=False, width=12)
        self._colorbar.setImageItem(self._image, insert_in=self.getPlotItem())

    def set_levels(self, lo: float, hi: float) -> None:
        self._levels = (lo, hi)
        self._colorbar.setLevels((lo, hi))

    def update_image(self, snapshot: np.ndarray) -> None:
        if self.auto_levels:
            lo, hi = np.percentile(snapshot, [5.0, 99.7])
            # smooth so the colours do not flicker
            lo = 0.9 * self._levels[0] + 0.1 * float(lo)
            hi = 0.9 * self._levels[1] + 0.1 * max(float(hi), float(lo) + 20.0)
            self.set_levels(lo, hi)
        self._image.setImage(snapshot, autoLevels=False, levels=self._levels)
        if snapshot.shape != self._shape:
            self._shape = snapshot.shape
            self._image.setRect(self._rect)

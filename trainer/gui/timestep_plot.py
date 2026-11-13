"""Live, deterministic probability preview; no sampling or training RNG consumption."""
import math
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from ..training.flow import FlowConfig, get_lin_function, sample_timesteps
from ..training.moss_timesteps import probability_table
from .widgets import THEME


from ..ui_core.timestep import Distribution, distribution, SAMPLER_KEYS


class _Canvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.error = ""
        self.setMinimumSize(260, 220)
        self.setMouseTracking(True)

    def graph_rect(self):
        return QtCore.QRectF(45, 27, max(1, self.width() - 60), max(1, self.height() - 72))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QColor(THEME.text_muted))
        if self.result is None:
            painter.drawText(self.rect().adjusted(8, 8, -8, -8),
                             QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
                             self.error or "Select timestep settings to preview")
            return
        r = self.graph_rect()
        values = self.result.probabilities
        ceiling = max(values) * 100 * 1.1
        painter.drawText(QtCore.QRectF(0, 0, self.width(), 23),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Probability per bin (%)")
        for i in range(5):
            y = r.bottom() - i * r.height() / 4
            painter.setPen(QtGui.QColor(THEME.text_muted))
            painter.drawText(QtCore.QRectF(0, y - 9, 39, 18), QtCore.Qt.AlignmentFlag.AlignRight,
                             f"{ceiling * i / 4:.1f}")
            pen = QtGui.QColor(THEME.text_muted)
            pen.setAlpha(50)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(r.left(), y), QtCore.QPointF(r.right(), y))
        width = r.width() / len(values)
        for i, probability in enumerate(values):
            height = probability * 100 / ceiling * r.height()
            painter.fillRect(QtCore.QRectF(r.left() + i * width, r.bottom() - height,
                                           max(.5, width - 1), height), QtGui.QColor(THEME.accent))
        painter.setPen(QtGui.QColor(THEME.text_muted))
        for i in range(6):
            x = r.left() + i * r.width() / 5
            painter.drawText(QtCore.QRectF(x - 20, r.bottom() + 3, 40, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, f"{i / 5:.1f}")
        painter.drawText(QtCore.QRectF(0, self.height() - 22, self.width(), 22),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Clean (t=0)  →  Pure noise (t=1)")

    def mouseMoveEvent(self, event):
        r = self.graph_rect()
        if self.result and r.contains(event.position()):
            n = len(self.result.probabilities)
            i = min(n - 1, int((event.position().x() - r.left()) / r.width() * n))
            right = "]" if i == n - 1 else ")"
            QtWidgets.QToolTip.showText(event.globalPosition().toPoint(),
                f"t ∈ [{i/n:.3f}, {(i+1)/n:.3f}{right}: "
                f"{100*self.result.probabilities[i]:.3f}% of draws", self)
        else:
            QtWidgets.QToolTip.hideText()


class TimestepDistributionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._flat = {}
        self._last = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.grid_controls = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.grid_controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("Preview latent H × W"))
        self.height_spin, self.width_spin = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin in (self.height_spin, self.width_spin):
            spin.setRange(1, 512)
            spin.setValue(64)
            spin.valueChanged.connect(lambda _: self.set_config(self._flat))
            row.addWidget(spin)
        self.grid_controls.setToolTip("Preview only; does not change training resolution. "
                                     "Resolution-aware shift uses the actual latent grid for each training image.")
        layout.addWidget(self.grid_controls)
        self.canvas = _Canvas()
        layout.addWidget(self.canvas)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        note = QtWidgets.QLabel("Sampling frequency, not loss weight. Before curriculum range mapping. "
                                "Hover a bar for its probability; vertical axis auto-scales.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def set_config(self, flat):
        self._flat = flat
        options = {key: flat[f"flow.{key}"] for key in SAMPLER_KEYS if f"flow.{key}" in flat}
        h, w = self.height_spin.value(), self.width_spin.value()
        signature = (tuple(options.items()), h, w)
        if signature == self._last:
            return
        self._last = signature
        self.grid_controls.setVisible(bool(options.get("flux_shift") or options.get("cabal_shift")))
        try:
            result = distribution(FlowConfig(**options), h, w)
            self.canvas.result = result
            self.canvas.error = ""
            self.summary.setText(f"Median t: {result.median:.3f}   •   "
                                 f"High noise (t ≥ 0.9): {result.high_noise:.1%}   •   Shift: {result.shift:.3g}")
        except (ValueError, TypeError, OverflowError) as exc:
            self.canvas.result = None
            self.canvas.error = str(exc)
            self.summary.setText("Fix the timestep settings to update the preview.")
        self.canvas.update()

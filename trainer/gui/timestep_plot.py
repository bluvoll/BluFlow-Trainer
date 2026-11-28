"""Live, deterministic probability preview; no sampling or training RNG consumption."""
from PySide6 import QtCore, QtGui, QtWidgets

from ..training.flow import FlowConfig
from ..ui_core.timestep import (
    Distribution,
    SAMPLER_KEYS,
    distribution,
    resolution_distributions,
)
from .widgets import THEME

_PALETTE = ("#b49ddf", "#76c4bb", "#d5b27c", "#759edb", "#ee8d98", "#b4c67a")


def _curve_color(index: int) -> QtGui.QColor:
    if index < len(_PALETTE):
        return QtGui.QColor(_PALETTE[index])
    return QtGui.QColor.fromHsl(int((265 + index * 137.5) % 360), 140, 173)


class _Canvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.curves: list[dict] = []
        self._plotted: list[tuple[float, float, int, dict]] = []
        self.error = ""
        self.setMinimumSize(260, 240)
        self.setMouseTracking(True)

    def graph_rect(self):
        return QtCore.QRectF(45, 27, max(1, self.width() - 60), max(1, self.height() - 72))

    def _paint_isometric(self, painter: QtGui.QPainter):
        w, h = float(self.width()), float(self.height())
        left = 44.0
        bottom = h - 38.0
        depth_w = w * 0.2
        depth_h = min(54.0, max(24.0, h * 0.2))
        plot_w = max(1.0, w - left - depth_w - 16.0)
        plot_h = max(1.0, h - 102.0)
        ceiling = max(
            1e-8,
            *(p for c in self.curves for p in c["probabilities"]),
        ) * 1.1

        def project(t: float, depth: float, prob: float) -> QtCore.QPointF:
            return QtCore.QPointF(
                left + t * plot_w + depth * depth_w,
                bottom - depth * depth_h - (prob / ceiling) * plot_h,
            )

        grid_pen = QtGui.QColor(THEME.text_muted)
        grid_pen.setAlpha(60)
        muted = QtGui.QColor(THEME.text_muted)

        for i in range(5):
            prob = ceiling * i / 4.0
            painter.setPen(grid_pen)
            painter.drawLine(project(0, 0, prob), project(0, 1, prob))
            painter.drawLine(project(0, 1, prob), project(1, 1, prob))
            lbl = project(0, 0, prob)
            painter.setPen(muted)
            painter.drawText(
                QtCore.QRectF(0, lbl.y() - 9, left - 6, 18),
                QtCore.Qt.AlignmentFlag.AlignRight,
                f"{prob * 100:.2g}",
            )
            t = i / 4.0
            tick = project(t, 0, 0)
            painter.setPen(grid_pen)
            painter.drawLine(tick, project(t, 1, 0))
            painter.setPen(muted)
            painter.drawText(
                QtCore.QRectF(tick.x() - 20, tick.y() + 3, 40, 18),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"{t:.2f}",
            )

        painter.setPen(muted)
        painter.drawText(QtCore.QRectF(4, 2, w - 8, 20), QtCore.Qt.AlignmentFlag.AlignLeft, "Probability / bin (%)")
        painter.drawText(
            QtCore.QRectF(left, h - 20, plot_w, 18),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "Clean (t=0)  →  Pure noise (t=1)",
        )

        plotted: list[tuple[float, float, int, dict]] = []
        n_curves = len(self.curves)
        for index in range(n_curves - 1, -1, -1):
            curve = self.curves[index]
            depth = index / (n_curves - 1) if n_curves > 1 else 0.5
            painter.setPen(grid_pen)
            painter.drawLine(project(0, depth, 0), project(1, depth, 0))
            col = _curve_color(index)
            pen = QtGui.QPen(col, 1.8)
            painter.setPen(pen)
            path = QtGui.QPainterPath()
            probs = curve["probabilities"]
            n_bins = len(probs)
            for b, prob in enumerate(probs):
                pt = project((b + 0.5) / n_bins, depth, prob)
                if b == 0:
                    path.moveTo(pt)
                else:
                    path.lineTo(pt)
                plotted.append((pt.x(), pt.y(), b, curve))
            painter.drawPath(path)
            tick = project(1, depth, 0)
            painter.setPen(col)
            painter.drawText(
                QtCore.QRectF(tick.x() - 60, tick.y() - 18, 60, 16),
                QtCore.Qt.AlignmentFlag.AlignRight,
                f"{curve['resolution']}px",
            )
        self._plotted = plotted

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QColor(THEME.text_muted))
        if self.curves:
            self._paint_isometric(painter)
            return
        self._plotted = []
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
        if self.curves and self._plotted:
            px, py = event.position().x(), event.position().y()
            nearest = None
            best = 144.0
            for x, y, b, curve in self._plotted:
                d = (x - px) ** 2 + (y - py) ** 2
                if d < best:
                    best = d
                    nearest = (b, curve)
            if nearest is not None:
                b, curve = nearest
                n = len(curve["probabilities"])
                right = "]" if b == n - 1 else ")"
                QtWidgets.QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"{curve['resolution']}px · shift {curve['shift']:.3f}\n"
                    f"t ∈ [{b/n:.2f}, {(b+1)/n:.2f}{right}: "
                    f"{100 * curve['probabilities'][b]:.3f}%",
                    self,
                )
            else:
                QtWidgets.QToolTip.hideText()
            return
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
                                "Hover a bar or curve for its probability; vertical axis auto-scales.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def set_config(self, flat):
        self._flat = flat
        options = {key: flat[f"flow.{key}"] for key in SAMPLER_KEYS if f"flow.{key}" in flat}
        h, w = self.height_spin.value(), self.width_spin.value()
        res_sig = (
            flat.get("dataset.resolution"),
            tuple(flat.get("dataset.resolutions") or ()),
            bool(flat.get("resolution_schedule.enabled")),
            tuple(s.get("resolution") for s in (flat.get("resolution_schedule.stages") or ()) if isinstance(s, dict)),
        )
        signature = (tuple(options.items()), h, w, res_sig)
        if signature == self._last:
            return
        self._last = signature
        dynamic = bool(options.get("flux_shift") or options.get("cabal_shift") or options.get("res_shift"))
        self.grid_controls.setVisible(dynamic)
        try:
            cfg = FlowConfig(**options)
            curves = resolution_distributions(cfg, flat) if dynamic else []
            result = distribution(cfg, h, w)
            self.canvas.curves = curves
            self.canvas.result = result
            self.canvas.error = ""
            if curves:
                legend = "   •   ".join(
                    f"{c['resolution']}px: shift {c['shift']:.3f} (med {c['median']:.3f})"
                    for c in curves
                )
                self.summary.setText(legend)
            else:
                self.summary.setText(f"Median t: {result.median:.3f}   •   "
                                     f"High noise (t ≥ 0.9): {result.high_noise:.1%}   •   Shift: {result.shift:.3g}")
        except (ValueError, TypeError, OverflowError) as exc:
            self.canvas.curves = []
            self.canvas.result = None
            self.canvas.error = str(exc)
            self.summary.setText("Fix the timestep settings to update the preview.")
        self.canvas.update()

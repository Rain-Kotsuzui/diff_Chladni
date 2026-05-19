"""
Main GUI Application for Chladni Pattern Simulator.
PySide6 + Matplotlib backend.
"""
import sys
import numpy as np

# Ensure Qt backend before importing pyplot
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QPushButton,
    QComboBox, QSplitter, QScrollArea, QFrame, QFileDialog, QMessageBox,
    QButtonGroup, QRadioButton, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer

from gui_solver import SimulationController
from brush_engine import (
    generate_stamp, apply_brush,
    save_preset, load_preset, list_presets, delete_preset, default_preset,
    ensure_presets_dir,
)


class ChladniGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chladni Pattern Simulator")
        self.resize(1400, 950)

        self.controller = SimulationController(target_N=128, preview_N=48)
        self.controller.result_ready.connect(self.on_result_ready)
        self.controller.status_changed.connect(self.on_status_changed)
        self.controller.error_occurred.connect(self.on_error)

        # Brush state
        self.brush_mode = "view"  # 'view', 'raise', 'lower'
        self.brush_size = 15
        self.brush_hardness = 0.5
        self.brush_flow = 0.5
        self.brush_stamp = generate_stamp(self.brush_size, self.brush_hardness)
        self._mouse_pressed = False
        self._last_brush_pos = None

        self._build_ui()
        self._refresh_presets()

        # Initial preview
        QTimer.singleShot(100, self.controller._do_preview)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(420)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setAlignment(Qt.AlignTop)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # Parameter Group
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)

        self.param_widgets = {}
        params_def = [
            ("freq", "Frequency (Hz)", 10.0, 5000.0, 750.0, 1.0, True),
            ("N", "Grid Resolution N", 16, 512, 128, 1, False),
            ("L", "Plate Size L (m)", 0.01, 1.0, 0.2, 0.001, True),
            ("E", "Young's Modulus (Pa)", 1e9, 200e9, 70e9, 1e9, True),
            ("rho", "Density (kg/m³)", 100.0, 20000.0, 2700.0, 10.0, True),
            ("nu", "Poisson's Ratio", 0.0, 0.5, 0.33, 0.01, True),
            ("eta", "Damping Coefficient", 0.0, 1e-2, 1e-4, 1e-5, True),
        ]

        for key, label, min_v, max_v, init_v, step, is_float in params_def:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(140)
            row.addWidget(lbl)

            if is_float:
                spin = QDoubleSpinBox()
                spin.setDecimals(6 if key == "eta" else (2 if key in ("L", "nu") else 1))
                spin.setSingleStep(step)
                spin.setRange(min_v, max_v)
                spin.setValue(init_v)
                spin.setReadOnly(False)
                spin.setKeyboardTracking(True)
                spin.setFocusPolicy(Qt.StrongFocus)
                spin.setAlignment(Qt.AlignRight)
                spin.setMinimumWidth(90)
                spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)

                def fmt_val(val, dec):
                    if abs(val) >= 1e7 or (abs(val) < 1e-3 and val != 0):
                        return f"{val:.3e}"
                    return f"{val:.{dec}f}"

                val_lbl = QLabel(fmt_val(init_v, spin.decimals()))
                val_lbl.setMinimumWidth(70)
                val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 10000)
                slider.setMinimumWidth(60)
                # Map float range to int slider
                def make_mappers(k, s, mn, mx, sp, vl):
                    def slider_to_spin(v):
                        ratio = v / 10000.0
                        val = mn + ratio * (mx - mn)
                        # snap to step
                        val = round(val / sp) * sp
                        s.blockSignals(True)
                        s.setValue(val)
                        s.blockSignals(False)
                        vl.setText(fmt_val(val, s.decimals()))
                        self._param_changed(k, val)
                    def spin_to_slider(v):
                        ratio = (v - mn) / (mx - mn)
                        iv = int(ratio * 10000)
                        sld = self.param_widgets[k]["slider"]
                        sld.blockSignals(True)
                        sld.setValue(iv)
                        sld.blockSignals(False)
                        vl.setText(fmt_val(v, s.decimals()))
                        self._param_changed(k, v)
                    return slider_to_spin, spin_to_slider
                s2s, p2s = make_mappers(key, spin, min_v, max_v, step, val_lbl)
                slider.valueChanged.connect(s2s)
                spin.valueChanged.connect(p2s)
                row.addWidget(slider)
                row.addWidget(val_lbl)
                row.addWidget(spin)
                self.param_widgets[key] = {"spin": spin, "slider": slider, "val_lbl": val_lbl}
            else:
                spin = QSpinBox()
                spin.setRange(min_v, max_v)
                spin.setSingleStep(step)
                spin.setValue(init_v)
                spin.setReadOnly(False)
                spin.setKeyboardTracking(True)
                spin.setFocusPolicy(Qt.StrongFocus)
                spin.setAlignment(Qt.AlignRight)
                spin.setMinimumWidth(70)
                spin.valueChanged.connect(lambda v, k=key: self._param_changed(k, v))
                row.addWidget(spin)
                self.param_widgets[key] = {"spin": spin}

            param_layout.addLayout(row)

        # Apply button
        self.apply_btn = QPushButton("Apply to Target Resolution")
        self.apply_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.apply_btn.clicked.connect(self.controller.request_apply)
        param_layout.addWidget(self.apply_btn)

        left_layout.addWidget(param_group)

        # Thickness Tools Group
        thick_group = QGroupBox("Thickness Tools")
        thick_layout = QVBoxLayout(thick_group)

        # Uniform fill
        uniform_row = QHBoxLayout()
        uniform_row.addWidget(QLabel("Uniform thickness (m):"))
        self.uniform_spin = QDoubleSpinBox()
        self.uniform_spin.setRange(0.00005, 0.01)
        self.uniform_spin.setDecimals(6)
        self.uniform_spin.setSingleStep(0.0001)
        self.uniform_spin.setValue(0.0008)
        self.uniform_spin.setReadOnly(False)
        self.uniform_spin.setKeyboardTracking(True)
        self.uniform_spin.setFocusPolicy(Qt.StrongFocus)
        self.uniform_spin.setAlignment(Qt.AlignRight)
        self.uniform_spin.setMinimumWidth(140)
        self.uniform_spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)
        uniform_row.addWidget(self.uniform_spin)
        self.uniform_btn = QPushButton("Set Uniform")
        self.uniform_btn.clicked.connect(self._set_uniform_thickness)
        uniform_row.addWidget(self.uniform_btn)
        thick_layout.addLayout(uniform_row)

        # Brush mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Brush Mode:"))
        self.mode_group = QButtonGroup(self)
        self.mode_view = QRadioButton("View")
        self.mode_raise = QRadioButton("Raise")
        self.mode_lower = QRadioButton("Lower")
        self.mode_view.setChecked(True)
        self.mode_group.addButton(self.mode_view)
        self.mode_group.addButton(self.mode_raise)
        self.mode_group.addButton(self.mode_lower)
        mode_row.addWidget(self.mode_view)
        mode_row.addWidget(self.mode_raise)
        mode_row.addWidget(self.mode_lower)
        self.mode_group.buttonClicked.connect(self._brush_mode_changed)
        thick_layout.addLayout(mode_row)

        left_layout.addWidget(thick_group)

        # Brush Settings Group
        brush_group = QGroupBox("Brush Settings")
        brush_layout = QVBoxLayout(brush_group)

        def make_slider_row(name, min_v, max_v, init_v, decimals=2):
            row = QHBoxLayout()
            row.addWidget(QLabel(name + ":"))
            sld = QSlider(Qt.Horizontal)
            sld.setRange(0, 1000)
            sld.setValue(int((init_v - min_v) / (max_v - min_v) * 1000))
            lbl = QLabel(f"{init_v:.{decimals}f}")
            row.addWidget(sld)
            row.addWidget(lbl)
            return row, sld, lbl, min_v, max_v

        r1, self.size_sld, self.size_lbl, _, _ = make_slider_row("Size", 1, 100, 15, 0)
        self.size_sld.valueChanged.connect(self._brush_size_changed)
        brush_layout.addLayout(r1)

        r2, self.hard_sld, self.hard_lbl, _, _ = make_slider_row("Hardness", 0.0, 1.0, 0.5, 2)
        self.hard_sld.valueChanged.connect(self._brush_hardness_changed)
        brush_layout.addLayout(r2)

        r3, self.flow_sld, self.flow_lbl, _, _ = make_slider_row("Flow", 0.0, 1.0, 0.5, 2)
        self.flow_sld.valueChanged.connect(self._brush_flow_changed)
        brush_layout.addLayout(r3)

        # Presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(120)
        preset_row.addWidget(self.preset_combo)
        self.load_preset_btn = QPushButton("Load")
        self.load_preset_btn.clicked.connect(self._load_preset)
        preset_row.addWidget(self.load_preset_btn)
        self.save_preset_btn = QPushButton("Save")
        self.save_preset_btn.clicked.connect(self._save_preset_dialog)
        preset_row.addWidget(self.save_preset_btn)
        self.del_preset_btn = QPushButton("Del")
        self.del_preset_btn.clicked.connect(self._delete_preset)
        preset_row.addWidget(self.del_preset_btn)
        brush_layout.addLayout(preset_row)

        left_layout.addWidget(brush_group)
        left_layout.addStretch()

        # Right panel: visualization
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(4)
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)
        splitter.setSizes([350, 1000])

        # Chladni Pattern
        self.fig1 = Figure(figsize=(6, 4), facecolor='black')
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor('black')
        self.im1 = self.ax1.imshow(
            np.zeros((128, 128)), cmap='magma', origin='lower',
            extent=[0, self.controller.L * 100, 0, self.controller.L * 100], vmin=0, vmax=1
        )
        self.ax1.set_title("Chladni Pattern (Nodes)", color='white')
        self.ax1.tick_params(colors='white')
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.canvas1)

        # Energy Field
        self.fig2 = Figure(figsize=(6, 4), facecolor='black')
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_facecolor('black')
        self.im2 = self.ax2.imshow(
            np.zeros((128, 128)), cmap='viridis', origin='lower',
            extent=[0, self.controller.L * 100, 0, self.controller.L * 100]
        )
        self.ax2.set_title("Energy Field", color='white')
        self.ax2.tick_params(colors='white')
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.canvas2)

        # Thickness
        self.fig3 = Figure(figsize=(6, 4), facecolor='black')
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_facecolor('black')
        self.im3 = self.ax3.imshow(
            np.zeros((128, 128)), cmap='plasma', origin='lower',
            extent=[0, self.controller.L * 100, 0, self.controller.L * 100]
        )
        self.ax3.set_title("Plate Thickness (mm) — Paintable", color='white')
        self.ax3.tick_params(colors='white')
        self.fig3.colorbar(self.im3, ax=self.ax3, orientation='vertical', fraction=0.046, pad=0.04)
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.canvas3)

        # Status bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        # Mouse events for thickness canvas
        self.canvas3.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas3.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas3.mpl_connect('button_release_event', self._on_mouse_release)

    # Parameter handling
    def _param_changed(self, key, value):
        self.controller.set_parameter(key, value)
        if key == "L":
            # Update image extents immediately
            Lcm = value * 100
            extent = [0, Lcm, 0, Lcm]
            for im in (self.im1, self.im2, self.im3):
                im.set_extent(extent)
            for ax in (self.ax1, self.ax2, self.ax3):
                ax.set_xlim(0, Lcm)
                ax.set_ylim(0, Lcm)
            for fig in (self.fig1, self.fig2, self.fig3):
                fig.canvas.draw_idle()

    def _set_uniform_thickness(self):
        val = self.uniform_spin.value()
        self.controller.h_master[:, :] = val
        self.controller.set_thickness(self.controller.h_master)
        # Immediate visual update
        self.im3.set_data(self.controller.h_master * 1000.0)
        self.canvas3.draw_idle()

    # Brush handling
    def _brush_mode_changed(self, btn):
        if btn == self.mode_view:
            self.brush_mode = "view"
        elif btn == self.mode_raise:
            self.brush_mode = "raise"
        elif btn == self.mode_lower:
            self.brush_mode = "lower"

    def _brush_size_changed(self, v):
        self.brush_size = int(1 + v / 1000.0 * 99)
        self.size_lbl.setText(str(self.brush_size))
        self.brush_stamp = generate_stamp(self.brush_size, self.brush_hardness)

    def _brush_hardness_changed(self, v):
        self.brush_hardness = v / 1000.0
        self.hard_lbl.setText(f"{self.brush_hardness:.2f}")
        self.brush_stamp = generate_stamp(self.brush_size, self.brush_hardness)

    def _brush_flow_changed(self, v):
        self.brush_flow = v / 1000.0
        self.flow_lbl.setText(f"{self.brush_flow:.2f}")

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax3 or self.brush_mode == "view":
            return
        if event.button != 1:
            return
        self._mouse_pressed = True
        self._last_brush_pos = None
        self._paint_at(event.xdata, event.ydata)

    def _on_mouse_move(self, event):
        if not self._mouse_pressed:
            return
        if event.inaxes != self.ax3 or event.xdata is None or event.ydata is None:
            return
        # Interpolate between last position and current for smooth strokes
        if self._last_brush_pos is not None:
            x0, y0 = self._last_brush_pos
            x1, y1 = event.xdata, event.ydata
            dist = np.hypot(x1 - x0, y1 - y0)
            step = self.controller.L * 100 / self.controller.target_N * 0.5
            if dist > step:
                n = int(dist / step) + 1
                for i in range(1, n + 1):
                    t = i / n
                    self._paint_at(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        self._paint_at(event.xdata, event.ydata)
        self._last_brush_pos = (event.xdata, event.ydata)

    def _on_mouse_release(self, event):
        self._mouse_pressed = False
        self._last_brush_pos = None

    def _refresh_thickness_display(self, h_mm: np.ndarray):
        """Update thickness image with consistent clim based on master data."""
        self.im3.set_data(h_mm)
        # Always use h_master range for consistent color mapping
        h_master_mm = self.controller.h_master * 1000.0
        vmin = float(h_master_mm.min())
        vmax = float(h_master_mm.max())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1e-9
        self.im3.set_clim(vmin, vmax)
        self.canvas3.draw_idle()

    def _paint_at(self, xdata, ydata):
        if xdata is None or ydata is None:
            return
        N = self.controller.target_N
        L_cm = self.controller.L * 100
        # Map from cm to array index
        col = int((xdata / L_cm) * N)
        row = int((ydata / L_cm) * N)
        if not (0 <= row < N and 0 <= col < N):
            return

        h = self.controller.get_master_thickness()
        apply_brush(
            h, row, col,
            mode=self.brush_mode,
            flow=self.brush_flow,
            stamp=self.brush_stamp,
            delta_h=0.0001,
        )
        self.controller.set_master_thickness(h)
        self._refresh_thickness_display(h * 1000.0)

    # Preset management
    def _refresh_presets(self):
        self.preset_combo.clear()
        for name in list_presets():
            self.preset_combo.addItem(name)

    def _load_preset(self):
        name = self.preset_combo.currentText()
        if not name:
            return
        p = load_preset(name)
        self.brush_size = int(p.get("size", 15))
        self.brush_hardness = float(p.get("hardness", 0.5))
        self.brush_flow = float(p.get("flow", 0.5))
        mode = p.get("mode", "raise")

        self.size_sld.setValue(int((self.brush_size - 1) / 99 * 1000))
        self.hard_sld.setValue(int(self.brush_hardness * 1000))
        self.flow_sld.setValue(int(self.brush_flow * 1000))

        if mode == "raise":
            self.mode_raise.setChecked(True)
        elif mode == "lower":
            self.mode_lower.setChecked(True)
        else:
            self.mode_view.setChecked(True)
        self._brush_mode_changed(self.mode_group.checkedButton())
        self.brush_stamp = generate_stamp(self.brush_size, self.brush_hardness)

    def _save_preset_dialog(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name:
            mode = self.brush_mode
            save_preset(name, self.brush_size, self.brush_hardness, self.brush_flow, mode)
            self._refresh_presets()
            idx = self.preset_combo.findText(name)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)

    def _delete_preset(self):
        name = self.preset_combo.currentText()
        if not name:
            return
        delete_preset(name)
        self._refresh_presets()

    # Result handling
    def on_result_ready(self, pattern, energy, h_display, status, label):
        N = pattern.shape[0]
        Lcm = self.controller.L * 100
        extent = [0, Lcm, 0, Lcm]

        self.im1.set_data(pattern)
        self.im1.set_extent(extent)
        self.ax1.set_title(f"Chladni Pattern — {status}", color='white')

        self.im2.set_data(energy)
        self.im2.set_extent(extent)
        vmin = np.min(energy)
        vmax = np.max(energy)
        self.im2.set_clim(vmin, vmax)
        # Remove old contours and add new
        for c in self.ax2.collections:
            c.remove()
        if N >= 16:
            x_coords = np.linspace(0, Lcm, N)
            y_coords = np.linspace(0, Lcm, N)
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
            non_zero = energy[energy > 1e-18]
            if non_zero.size > 0:
                threshold = np.percentile(non_zero, 5)
                self.ax2.contour(X_grid, Y_grid, energy, levels=[threshold], colors='white', linewidths=1.5)

        self.im3.set_extent(extent)
        self._refresh_thickness_display(h_display)

        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
        self.canvas3.draw_idle()
        self.status_label.setText(status)

    def on_status_changed(self, text):
        self.status_label.setText(text)
        if "Solving full" in text:
            self.apply_btn.setEnabled(False)
        else:
            self.apply_btn.setEnabled(True)

    def on_error(self, msg):
        QMessageBox.critical(self, "Solver Error", msg)
        self.apply_btn.setEnabled(True)


def main():
    ensure_presets_dir()
    app = QApplication(sys.argv)
    # Optional dark palette
    app.setStyle('Fusion')
    win = ChladniGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

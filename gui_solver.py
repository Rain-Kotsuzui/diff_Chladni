"""
Simulation Controller for the Chladni GUI.
Encapsulates solve logic, preview/full-res modes, and background threading.
"""
import numpy as np
import warp as wp
import cv2
from scipy import ndimage

from PySide6.QtCore import QObject, QThread, Signal, QTimer

from parameter import PlateParams
from direct_solver import DifferentiableDirectSolver
from device_manager import WARP_DEVICE, init_devices

init_devices()


class SolveWorker(QThread):
    """Background worker for full-resolution solves."""
    result_ready = Signal(object, object, object, object, str)  # pattern, energy, h_display, status, label
    error_occurred = Signal(str)

    def __init__(self, solver: DifferentiableDirectSolver, params: PlateParams,
                 h_wp, fr_wp, fi_wp, N, L, label=""):
        super().__init__()
        self.solver = solver
        self.params = params
        self.h_wp = h_wp
        self.fr_wp = fr_wp
        self.fi_wp = fi_wp
        self.N = N
        self.L = L
        self.label = label

    def run(self):
        try:
            w_complex = self.solver.solve(self.h_wp, self.fr_wp, self.fi_wp)
            wr, wi = w_complex.real, w_complex.imag
            amp = np.sqrt(wr**2 + wi**2).reshape((self.N, self.N))

            clip_max = np.percentile(amp, 98)
            if clip_max < 1e-20:
                clip_max = 1e-20
            display_pattern = np.exp(-(amp / (clip_max * 0.15)) ** 2)

            energy_field = (wr ** 2.0 + wi ** 2.0).reshape(self.N, self.N)

            h_display = self.h_wp.numpy().reshape((self.N, self.N)) * 1000.0

            status = f"Ready (N={self.N})"
            self.result_ready.emit(display_pattern, energy_field, h_display, status, self.label)
        except Exception as e:
            self.error_occurred.emit(str(e))


class SimulationController(QObject):
    """Manages solver state, parameters, preview logic, and background solves."""
    result_ready = Signal(object, object, object, object, str)  # pattern, energy, h_display, status, label
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, target_N=128, preview_N=48, parent=None):
        super().__init__(parent)
        self.target_N = target_N
        self.preview_N = preview_N
        self.current_N = preview_N

        self.L = 0.2  # meters
        self.freq = 750.0

        self.params = PlateParams()
        self.params.E = 70e9
        self.params.nu = 0.33
        self.params.rho = 2700.0
        self.params.eta = 1e-4
        self._update_grid_params()

        self.solver = DifferentiableDirectSolver(self.params, self.preview_N, self.preview_N)

        # Master thickness field at target resolution
        self.h_master = np.ones((self.target_N, self.target_N), dtype=np.float32) * 0.0008

        # Force: single center source (can be extended)
        self.fr_master = np.zeros(self.target_N * self.target_N, dtype=np.float32)
        idx1 = (self.target_N // 2) * self.target_N + (self.target_N // 2)
        self.fr_master[idx1] = 1.0
        self.fi_master = np.zeros(self.target_N * self.target_N, dtype=np.float32)

        # Debounce timer for preview
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._do_preview)

        self._worker = None
        self._pending_preview = False

    def _update_grid_params(self):
        self.params.omega = 2.0 * np.pi * self.freq
        self.params.dx = self.L / self.current_N
        self.params.dy = self.L / self.current_N

    def set_parameter(self, key: str, value):
        """Update a simulation parameter and trigger preview debounce."""
        changed = False
        if key == "freq":
            if self.freq != float(value):
                self.freq = float(value)
                changed = True
        elif key == "N":
            val = int(value)
            if self.target_N != val:
                # Resize master thickness field
                if val != self.target_N:
                    self.h_master = cv2.resize(self.h_master, (val, val), interpolation=cv2.INTER_LINEAR)
                    self.fr_master = np.zeros(val * val, dtype=np.float32)
                    idx1 = (val // 2) * val + (val // 2)
                    self.fr_master[idx1] = 1.0
                    self.fi_master = np.zeros(val * val, dtype=np.float32)
                self.target_N = val
                changed = True
        elif key == "L":
            if self.L != float(value):
                self.L = float(value)
                changed = True
        elif key == "E":
            if self.params.E != float(value):
                self.params.E = float(value)
                changed = True
        elif key == "rho":
            if self.params.rho != float(value):
                self.params.rho = float(value)
                changed = True
        elif key == "nu":
            if self.params.nu != float(value):
                self.params.nu = float(value)
                changed = True
        elif key == "eta":
            if self.params.eta != float(value):
                self.params.eta = float(value)
                changed = True

        if changed:
            self._debounce_timer.stop()
            self._debounce_timer.start(300)  # 300 ms debounce

    def set_thickness(self, h_2d: np.ndarray):
        """Update master thickness field from external editing."""
        if h_2d.shape != self.h_master.shape:
            # If shape mismatch (e.g., after N change), resize
            h_2d = cv2.resize(h_2d, (self.target_N, self.target_N), interpolation=cv2.INTER_LINEAR)
        self.h_master = h_2d.astype(np.float32)
        self._debounce_timer.stop()
        self._debounce_timer.start(300)

    def _prepare_arrays(self, N: int):
        """Downsample master arrays to target N and create Warp arrays."""
        if N == self.target_N:
            h_2d = self.h_master
            fr_1d = self.fr_master
            fi_1d = self.fi_master
        else:
            h_2d = cv2.resize(self.h_master, (N, N), interpolation=cv2.INTER_LINEAR)
            fr_1d = np.zeros(N * N, dtype=np.float32)
            idx1 = (N // 2) * N + (N // 2)
            fr_1d[idx1] = 1.0
            fi_1d = np.zeros(N * N, dtype=np.float32)

        h_wp = wp.from_numpy(h_2d.flatten(), device=WARP_DEVICE)
        fr_wp = wp.from_numpy(fr_1d, device=WARP_DEVICE)
        fi_wp = wp.from_numpy(fi_1d, device=WARP_DEVICE)
        return h_wp, fr_wp, fi_wp, h_2d

    def _do_preview(self):
        """Run a low-resolution preview solve synchronously."""
        if self._worker is not None and self._worker.isRunning():
            self.status_changed.emit("Busy (full solve running)...")
            self._pending_preview = True
            return
        self._pending_preview = True
        self.status_changed.emit(f"Previewing (N={self.preview_N})...")
        N = min(self.target_N, self.preview_N)
        self.current_N = N
        self._update_grid_params()
        self.solver.resize(N, N)

        h_wp, fr_wp, fi_wp, h_2d = self._prepare_arrays(N)
        try:
            w_complex = self.solver.solve(h_wp, fr_wp, fi_wp)
            wr, wi = w_complex.real, w_complex.imag
            amp = np.sqrt(wr**2 + wi**2).reshape((N, N))

            clip_max = np.percentile(amp, 98)
            if clip_max < 1e-20:
                clip_max = 1e-20
            display_pattern = np.exp(-(amp / (clip_max * 0.15)) ** 2)

            energy_field = (wr ** 2.0 + wi ** 2.0).reshape(N, N)
            h_display = h_2d * 1000.0

            self._pending_preview = False
            self.status_changed.emit(f"Preview (N={N})")
            self.result_ready.emit(display_pattern, energy_field, h_display, f"Preview (N={N})", "preview")
        except Exception as e:
            self._pending_preview = False
            self.error_occurred.emit(f"Preview error: {e}")

    def request_apply(self):
        """Run a full-resolution solve in a background thread."""
        if self._worker is not None and self._worker.isRunning():
            # Wait or ignore? For simplicity, ignore new request until current finishes.
            self.status_changed.emit("Busy, please wait...")
            return

        self.status_changed.emit(f"Solving full resolution (N={self.target_N})...")
        self.current_N = self.target_N
        self._update_grid_params()
        self.solver.resize(self.target_N, self.target_N)

        h_wp, fr_wp, fi_wp, h_2d = self._prepare_arrays(self.target_N)
        self._worker = SolveWorker(self.solver, self.params, h_wp, fr_wp, fi_wp,
                                   self.target_N, self.L, label="full")
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.start()

    def _on_worker_result(self, pattern, energy, h_display, status, label):
        self.status_changed.emit(status)
        self.result_ready.emit(pattern, energy, h_display, status, label)

    def _on_worker_error(self, msg):
        self.status_changed.emit(f"Error: {msg}")
        self.error_occurred.emit(msg)

    def _cleanup_worker(self):
        self._worker = None
        if self._pending_preview:
            self._do_preview()

    def get_master_thickness(self) -> np.ndarray:
        """Return the current master thickness field (meters)."""
        return self.h_master.copy()

    def set_master_thickness(self, h_2d: np.ndarray):
        """Set master thickness field directly."""
        self.h_master = h_2d.astype(np.float32)
        self._debounce_timer.stop()
        self._debounce_timer.start(300)

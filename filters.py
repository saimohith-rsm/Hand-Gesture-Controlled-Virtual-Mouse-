"""filters.py
One‑Euro filter implementation with deadzone jitter suppression and real dt support.
"""

from typing import Tuple, Optional
import math

class OneEuroFilter:
    """One‑Euro filter for low‑latency, jitter-free cursor smoothing.

    Parameters
    ----------
    freq : float
        Default sampling frequency (used if real dt is not passed).
    min_cutoff : float
        Minimum cutoff frequency (lower = smoother when hovering).
    beta : float
        Speed coefficient (higher = less lag on fast sweeps).
    d_cutoff : float
        Cutoff frequency for the derivative filter.
    deadzone : float
        Movement radius in pixels below which jitter is completely suppressed.
    """

    def __init__(self,
                 freq: float = 30.0,
                 min_cutoff: float = 0.5,
                 beta: float = 0.2,
                 d_cutoff: float = 1.0,
                 deadzone: float = 1.5):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.deadzone = float(deadzone)
        self.x_prev = None  # type: ignore
        self.dx_prev = (0.0, 0.0)

    def reset(self):
        """Reset internal filter state when tracking is lost so cursor does not jump on reappearance."""
        self.x_prev = None
        self.dx_prev = (0.0, 0.0)

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """Compute smoothing factor α from cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: Tuple[float, float], dt: Optional[float] = None) -> Tuple[float, float]:
        """Filter a new 2‑D point with optional real dt (time since last frame)."""
        if self.x_prev is None:
            self.x_prev = x
            return x

        # Calculate frequency from real dt if provided
        freq = (1.0 / dt) if (dt is not None and dt > 0) else self.freq

        # Deadzone: eliminate camera sensor noise and natural finger tremors when hovering
        dist = math.hypot(x[0] - self.x_prev[0], x[1] - self.x_prev[1])
        if dist < self.deadzone:
            return self.x_prev

        # Derivative (velocity) estimation
        dx = (x[0] - self.x_prev[0], x[1] - self.x_prev[1])
        dx = (dx[0] * freq, dx[1] * freq)
        alpha_d = self._alpha(self.d_cutoff, freq)
        self.dx_prev = (
            alpha_d * dx[0] + (1 - alpha_d) * self.dx_prev[0],
            alpha_d * dx[1] + (1 - alpha_d) * self.dx_prev[1],
        )

        # Adaptive cutoff based on speed (slow = smooth, fast = instant)
        speed = math.hypot(self.dx_prev[0], self.dx_prev[1])
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, freq)
        x_filt = (
            alpha * x[0] + (1 - alpha) * self.x_prev[0],
            alpha * x[1] + (1 - alpha) * self.x_prev[1],
        )
        self.x_prev = x_filt
        return x_filt

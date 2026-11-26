"""radar_bo: constrained BO, planar EKF tracking and radar resource allocation research framework.

Research software for studying constrained Bayesian optimization, statistical
state estimation and resource allocation in a simulated civilian UAV corridor
monitoring scenario. Synthetic, non-deployable; see README.md for the full
specification and current limitations.

Layer separation (README section 3, non-negotiable):

* ``radar_bo.config`` and the simulation layer must NOT import the optimizer.
* The generic optimizer (``radar_bo.optimization``) must NOT depend on radar
  equations.

Units: SI internally, radians for angles, linear (not dB) quantities, double
precision for covariance and GP operations.
"""

__version__ = "0.1.0"

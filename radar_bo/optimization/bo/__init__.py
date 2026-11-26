"""Generic Bayesian-optimization engine (M5).

Application-independent infrastructure: optimization problem interface,
synthetic mathematical benchmarks, independent reference solvers, dataset
management, NumPy<->PyTorch conversion, GP surrogate, LogExpectedImprovement
acquisition, candidate generation, and the generic single-candidate BO loop
with budget accounting and stopping conditions.

Layer separation (M5 instruction 30): this subpackage MAY import
PyTorch/GPyTorch/BoTorch; it must NOT import radar, environment or simulation
modules (guarded by tests). Radar budget semantics are not used here — M5
synthetic problems carry their own explicitly defined instantaneous
mathematical constraints (M4 review 3.1).
"""

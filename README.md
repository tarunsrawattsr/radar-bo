# Cognitive Radar Resource Management — AI Agent Implementation Guide

## 1. Mission and scope

Build a reproducible Python research software framework for studying constrained Bayesian optimization (BO), statistical state estimation, and resource allocation in a simulated civilian UAV corridor monitoring scenario. Use Python, Pydantic, NumPy, PyTorch, GPyTorch and BoTorch. The framework must separate the simulation, objective, optimization, and experimental analysis layers.

This document is an instruction specification, not source code. Implement incrementally, test each milestone before proceeding, and document assumptions. The initial deliverable is a validated planar state-estimation baseline and a generic constrained BO research harness tested on synthetic objectives. Application-specific power-dependent sensing and corridor detection models remain research extensions requiring independently validated physical assumptions; do not treat placeholders as validated models or deployable surveillance capabilities.

### Research foundations

- Z. Lu et al., *Learning-Based Cognitive Radar Resource Management for Scanning and Multi-Target Tracking*, IEEE ICC 2024, DOI: 10.1109/ICC51166.2024.10622225. Use the supplied paper for its motion model, EKF, covariance-based cost, scanning metric, and simulation parameters.
- *Designing Cellular Networks for UAV Corridors via Bayesian Optimization*, IEEE GLOBECOM 2023. Use the methodological concepts discussed in the project: Gaussian process surrogate, expected improvement, complete configuration observations, and block-wise acquisition search. Independently check the original paper before claiming exact reproduction of its equations or algorithm.

### Non-negotiable agent instructions

- Never invent missing paper parameters. Preserve unknown values as `None`, with a provenance record. Fail clearly when an experiment requires an unresolved value.
- Distinguish paper-reported values, standard constants, proposed assumptions, and experimental settings.
- Use SI units internally, radians for angular computations, and linear quantities where required by the mathematical model.
- Keep true simulated states inaccessible to the optimizer and estimator except through explicitly modeled observations; truth is for simulation and evaluation only.
- Separate physical simulation time from BO iteration count. Candidate evaluation must not mutate the main simulation state.
- Do not claim that BO is superior, globally optimal, or more efficient before collecting comparative evidence.
- Do not fabricate experimental results or fill missing plots with illustrative data presented as measurements.
- Use synthetic objective functions and synthetic observation-quality settings until required physical assumptions are independently justified.
- Record deviations from the reference papers, including initialization, covariance interpretation, objective horizon and constraint semantics.
- Keep a progress checklist, report failed tests honestly, and save reproducible experiment configurations.

## 2. Environment and dependencies

1. Create an isolated virtual environment and choose a Python version compatible with all installed dependencies.
2. Install Python scientific packages: NumPy, Pydantic (prefer version 2), PyTorch, GPyTorch and BoTorch. Add SciPy, Matplotlib, Pandas and Pytest for validation, numerical baselines, plots and testing.
3. Use CPU execution initially and double precision for NumPy covariance operations and GP tensors.
4. Record exact dependency versions in a reproducible lock or requirements file. Consult installed-version API documentation rather than assuming signatures.
5. Initialize version control. Exclude virtual environments, caches and bulky generated experiment outputs as appropriate.
6. Create a README documenting installation, test execution, experiment execution, and limitations.

Official documentation: [Pydantic](https://docs.pydantic.dev/), [NumPy](https://numpy.org/doc/), [PyTorch](https://docs.pytorch.org/), [GPyTorch](https://docs.gpytorch.ai/), [BoTorch](https://botorch.org/), [SciPy](https://docs.scipy.org/doc/scipy/), [Pytest](https://docs.pytest.org/).

## 3. Suggested repository structure

- `config/`: tracking_config.py, scanning_config.py, resource_config.py, environment_config.py, optimization_config.py, experiment_config.py, simulation_config.py.
- `environment/`: target_state.py, motion_model.py, simulation_environment.py, corridor.py (future scenario interface).
- `radar/`: measurement_model.py, measurement_noise.py, extended_kalman_filter.py, tracking_cost.py, scanning_model.py.
- `optimization/`: allocation.py, constraints.py, objective.py, dataset.py, surrogate.py, acquisition.py, candidate_generator.py, joint_bo.py, alternating_bo.py, coordinate_bo.py.
- `benchmarks/`: equal_allocation.py, random_allocation.py, dwell_only_bo.py, direct_optimizer.py.
- `simulation/`: simulation_state.py, simulator.py, simulation_runner.py.
- `experiments/`: static_experiment.py, dynamic_experiment.py, comparison_experiment.py, sensitivity_experiment.py.
- `analysis/`: metrics.py, visualization.py, statistical_analysis.py.
- `tests/`: unit and integration tests corresponding to the modules above.
- `results/`: per-run configuration, datasets, logs, figures and metrics.
- Root: main.py, README.md, pyproject.toml, dependency lock file, parameter provenance register.

The names are recommendations, not mandatory architecture. Avoid unnecessary abstractions in the first prototype, but preserve dependency separation: configuration and simulation must not import the optimizer; the generic optimizer must not depend on radar equations.

## 4. Configuration layer (Pydantic)

### 4.1 TrackingConfig

| Field | Paper value | Validation or interpretation |
|---|---:|---|
| reference_range_variance | 100 m² | Positive; already a variance |
| reference_angle_variance | 0.0004 rad² | Positive; already a variance |
| maneuverability_noise | 25 | Paper notation ambiguous: Eq. (3) squares sigma_w, while prose and Table II describe variance. Document chosen interpretation and test sensitivity. |
| reference_distance | 3000 m | Positive |
| reference_dwell_time | 1 s | Positive |
| revisit_interval | 2.5 s | Positive |
| reference_snr | None | Positive linear value when resolved |
| reference_transmit_power | None | Extension, not paper default |
| state_dimension | 4 | Planar model |
| measurement_dimension | 2 | Range and azimuth |
| initial_state | zero vector | Paper-reported; singular measurement geometry requires an explicit initialization policy |
| initial_covariance | identity matrix, 4 x 4 | Symmetric, positive semidefinite |

Do not square variance fields a second time. Clearly distinguish a paper reproduction setting from an alternative numerical initialization.

### 4.2 ScanningConfig

Preserve the paper's scanning model interface and parameter definitions. The following values are unspecified in the paper: peak transmit power, transmit and receive antenna gains, wavelength, radar cross section, loss factor, system temperature, beam angular spacing, reference area, detection probability, false-alarm probability and required detection SNR. Store each as `None` initially. The source assumes full 360-degree azimuth coverage and a monostatic radar. Boltzmann's constant is a standard physical constant, not a fitted parameter. Detection-model assumptions are unresolved; the source refers to Shnidman's equation but does not provide every implementation setting.

Validate positive physical quantities, proper probability bounds, valid angular quantities and nonzero reference area when supplied. Allow a partially populated research configuration, but reject an attempt to execute an experiment requiring missing parameters.

### 4.3 ResourceConfig

- Paper dwell-time action set: 0.1 to 1.0 seconds in increments of 0.1.
- Paper normalized time-budget threshold: 1.
- Paper tracking/scanning tradeoff coefficient: 1e6; preserve for reproduction only, recalibrate if metric changes.
- Proposed power limits, reference power, scanning power and energy budget: `None` until justified.
- Explicitly distinguish average transmitted power from peak pulse power; do not silently treat them as equivalent.
- Validate bounds, grid consistency, resource-budget definitions and unit conventions.
- Document that the proposed hard per-slot budget is stronger than the source's discounted cumulative constraint.

### 4.4 EnvironmentConfig and ExperimentConfig

Record scenario ID, simulation dimension, number of synthetic targets, initial conditions, simulation duration, revisit interval, random seeds, motion model, arrival/departure metadata, evaluation budget, repetitions, output directory and experiment mode. Start with fixed-population synthetic planar scenarios. Treat civilian corridor geometry and variable population as later research features, not necessary to validate the optimizer.

### 4.5 BOConfig

| Field | Proposed initial choice |
|---|---|
| Initial observations | 30 (experimental) |
| Acquisition candidate pool | 500 (experimental) |
| GP | SingleTaskGP |
| Kernel | Explicit Matérn-5/2 |
| Acquisition | LogExpectedImprovement for low-noise baseline |
| Noisy objective option | qLogNoisyExpectedImprovement, if justified |
| Search modes | Joint, alternating, coordinate-wise |
| Input normalization | Enabled, consistent physical bounds |
| Outcome standardization | Enabled |
| Precision | Float64 |
| Evaluation budget, patience, tolerance, runtime budget | Configurable; no paper default |
| Random seed | Configurable |

### 4.6 Master configuration and provenance

Compose all nested models into `SimulationConfig`. Use Pydantic for validation and serialization, not numerical matrix computation. Maintain one authoritative source for shared fields such as revisit interval. Validate cross-model consistency. Save the exact serialized configuration with each run. Record parameter name, value, units, source, rationale and sensitivity range in a provenance register.

## 5. Simulation and statistical estimation baseline (NumPy)

### 5.1 State representation

Maintain distinct representations of true simulated state, prior estimate, posterior estimate, prior covariance, posterior covariance, observation and observation covariance. The original paper uses four state components (two positions and two velocities), a 4 x 4 state covariance and two measurement components with a 2 x 2 measurement covariance. Store identifiers and active status separately from the numerical state.

### 5.2 Motion model

Reproduce the source paper's constant-velocity transition and Gaussian process covariance from Equations (1)–(3) for the planar baseline. First test deterministic propagation on simple synthetic examples. Then test stochastic propagation with a documented process-noise interpretation. Check matrix dimensions, symmetry, positive semidefiniteness and reproducibility. Do not assume the paper's ambiguous noise notation has been resolved by the value 25 alone.

### 5.3 Measurement and EKF modules

Implement a standalone educational planar EKF baseline following the source paper's Equations (4), (7)–(13). Separate ideal observations, observation noise, local linearization, prediction and update. Test using synthetic measurement covariance inputs independent of resource allocation. Preserve angle-unit conventions and handle invalid geometry, including the source's zero-position initialization singularity. Prefer numerically stable covariance operations and verify validity of each update. Document any alternative initialization rather than attributing it to the source.

The measurement-quality interface must be interchangeable. The proposed dependence of measurement uncertainty on additional sensing resources is not yet a validated physical extension. Do not fabricate such a model to satisfy integration tests; use abstract synthetic settings instead.

### 5.4 Tracking metric

Implement the source's covariance-based metric, Equation (14), as the sum of posterior position variances. Return a scalar. Keep the metric independent of optimization, scanning and weights. Also record actual synthetic estimation error separately; an estimated covariance is not a guarantee of realized accuracy.

### 5.5 State snapshots and randomness

Define a simulation state that supports independent snapshots. Maintain separate seeded random streams for motion, observations, initial designs and acquisition candidate generation. Each counterfactual evaluation must begin from equivalent state and random assumptions. Preserve the main simulation state until a selected action is committed. Avoid leakage of ground truth into optimizer inputs.

## 6. Scanning metric interface

The original paper defines scanning performance as the ratio of theoretical detectable area to reference area, Equation (21). Preserve this definition in a standalone metric interface, with the physical model and detector assumptions handled separately. Do not clip this original ratio to one unless explicitly implementing a modified metric.

For the proposed civilian corridor application, the research report identifies normalized coverage fraction and average detection reliability as candidate metrics. Their definitions and physical assumptions are not finalized. Provide an interface and test it on synthetic normalized coverage or probability inputs. Do not implement or claim a validated corridor-specific beam schedule or detector from unresolved parameters. Keep any physical model extension distinct from the generic research harness.

Tests: reject negative resources, invalid probabilities and zero reference area; check valid output bounds for normalized metrics; preserve the original ratio's ability to exceed one; check missing-parameter errors and deterministic reproducibility.

## 7. Allocation, constraints and objective interfaces

### 7.1 Allocation representation

Create a generic structured allocation with two numerical blocks, consistent ordering, dimension validation and conversion to NumPy and PyTorch representations. In the research formulation the blocks correspond conceptually to dwell times and powers, giving a complete vector of dimension 2N for N objects. Use abstract synthetic variables to validate joint optimization before integrating any physical extensions.

### 7.2 Constraints

Implement a reusable constraint manager that checks variable bounds, allowed discrete values, shared resource constraints and vector structure. Return a structured result with validity and reasons for failure. Reuse this component in every optimizer and benchmark. For the synthetic baseline, filter candidates before acquisition scoring. Do not silently accept invalid vectors or assign arbitrary utility zero to failed evaluations.

### 7.3 Objective interface

The source's utility structure combines negative weighted estimation cost and positive scanning metric. Represent the objective as an interchangeable evaluator accepting a snapshot and complete feasible configuration, returning a scalar utility, component costs, feasibility and runtime. The optimizer must not know the internal equations. Use a synthetic constrained objective with an independently verifiable optimum for the first end-to-end integration.

### 7.4 Evaluation semantics

Separate three operations: initialize a state; evaluate a candidate without permanent mutation; advance the state after an accepted decision. Check that evaluating candidates in a different order does not accidentally change their starting state. Treat optimization iteration k and physical simulation time t as distinct indices.

## 8. Generic BoTorch infrastructure

### 8.1 Dataset

Each observation stores a complete input vector and corresponding observed scalar objective, plus optional metadata. Training input tensor shape: M x d; training output shape: M x 1. For an illustrative eight-dimensional synthetic problem with 30 samples, shapes are 30 x 8 and 30 x 1. Validate finite inputs and outputs, track duplicate samples, and distinguish deliberate noisy replications from accidental repeats. Preserve the exact configuration associated with each observation.

### 8.2 Initial design

Generate feasible configurations with random, Sobol or Latin hypercube sampling as appropriate to the synthetic problem. Evaluate each through the common evaluator. Initial values are observations, not known optima. Record rejection rates and seed. Include any starting incumbent in the dataset if it has been evaluated.

### 8.3 Conversion and scaling

Use NumPy for simulation and PyTorch float64 tensors for BO. Centralize conversions. Use `Normalize` and `Standardize` or equivalent documented transformations. Normalize only once, use consistent bounds, and check constraints in the correct coordinate system. Do not feed Pydantic objects directly to a GP.

### 8.4 Surrogate model

Use BoTorch `SingleTaskGP`, explicitly configure GPyTorch `MaternKernel` with nu=2.5 if reproducing the selected methodological choice, and fit using `ExactMarginalLogLikelihood` and `fit_gpytorch_mll`. Check installed-version APIs. The posterior must return finite means and nonnegative variances for valid inputs. Distinguish latent predictive uncertainty, observation noise, and physical estimation covariance. Test the surrogate independently on a known synthetic objective.

### 8.5 Acquisition and candidate selection

Use `LogExpectedImprovement` for the low-noise synthetic baseline, with the best observed objective as reference. Document that this conventional reference differs from the surrogate-based reference discussed in the UAV paper. For noisy objectives investigate the supported noisy EI variant. Construct valid candidate batches with appropriate tensor shapes; score them using the acquisition function; select the highest-scoring feasible candidate. A pool of 500 acquisition candidates does not mean 500 new objective evaluations. Consult installed-version documentation for acquisition batch dimensions and discrete optimization interfaces.

### 8.6 Generic optimization loop

1. Retrieve the dataset and fit or update the GP.
2. Construct the acquisition function from the current observations.
3. Generate and validate complete candidate vectors.
4. Score candidates and select one according to acquisition value.
5. Evaluate it through the objective interface.
6. Append the exact evaluated vector and finite observed utility.
7. Record the incumbent and diagnostics.
8. Check evaluation, iteration, patience, tolerance and runtime limits.

Return the best evaluated feasible configuration, not merely the highest predicted mean. For noisy objectives, verify promising solutions independently. Log failures instead of converting them into valid observations.

## 9. Optimization architectures — synthetic benchmark first

### 9.1 Joint BO

Implement joint acquisition search across all variables as the baseline. Reuse common dataset, surrogate, acquisition, candidate generation and feasibility modules. Validate against a known or independently verified optimum.

### 9.2 Alternating BO

Build a generic block-search abstraction. It must retain fixed coordinates, assemble complete vectors, validate feasibility, score the appropriate candidates and store complete observations. In a two-block synthetic problem, alternate optimization of the first block with the second fixed and the second block with the first fixed. Use one shared full-dimensional GP and dataset. Keep exact copies of evaluated configurations so that recorded utility always belongs to the correct vector. Track both the current iterate and best verified incumbent; acquisition-selected evaluations need not improve monotonically. Define acceptance and stopping rules explicitly.

### 9.3 Coordinate-wise BO

Reuse the block abstraction for smaller coordinate groups on synthetic problems. Full GP dimensionality does not decrease just because acquisition-search dimensionality decreases. Compare all three methods under equal objective-evaluation budgets, not equal outer-cycle counts. Do not claim block-wise convergence implies global optimality.

## 10. Benchmark and experiment manager

Provide equal allocation, random feasible search and direct numerical optimization or enumeration for appropriate synthetic problems. Use identical objective definitions, constraints, starting datasets, seed sets and evaluation budgets where applicable. Measure total runtime as well as sample efficiency. BO is not automatically appropriate for cheap analytical objectives.

Treat CDRL reproduction as a separate study. The source optimizes discounted utility and a discounted budget constraint; the initial BO baseline optimizes a single evaluation under hard feasibility. These are distinct problems. Do not claim a fair comparison without aligning assumptions and horizons.

Support fixed-state and dynamic experiment modes as separate interfaces. Do not pool observations from different contexts into a static GP without modeling the context. Variable target counts require a separately validated representation; do not silently pad or truncate observations. Contextual BO remains a future research item.

## 11. Logging and experimental outputs

Each evaluation record should contain experiment ID, method, random seed, physical time index if applicable, BO iteration, outer cycle, complete configuration, observed utility, objective components, feasibility, evaluator runtime, GP fitting time, acquisition time and best observed utility. Keep rich metadata separate from numerical training tensors.

Save a per-run configuration, dependency versions, observation dataset, evaluation history, failure logs and figures. Use unique result directories. Record exceptions, numerical warnings, invalid candidates and rejected evaluations. Only finite valid observations should enter the ordinary GP dataset.

For the generic BO baseline generate best verified objective versus evaluations, runtime decomposition, variability across seeds, feasibility statistics and scaling versus synthetic problem dimension. Use measured data only. For source-paper estimation reproduction, record true and estimated synthetic trajectories and estimation error. The paper's five figures are a reference for presentation, not evidence of our results: trajectories, distances, weights, time budget and utility comparisons.

Run multiple seeds, report mean/median, dispersion and appropriate confidence intervals. Use separate validation seeds and avoid selecting hyperparameters solely from favorable test runs. Investigate initial-design size, candidate-pool size, kernel, noise, stopping tolerance and evaluation budget.

## 12. Testing and acceptance criteria

| Component | Required tests |
|---|---|
| Configurations | Invalid values rejected; missing required values fail explicitly; serialization round trip |
| Motion | Known deterministic propagation; valid covariance; reproducibility |
| Measurement/EKF | Dimensions, covariance validity, invalid geometry handling, consistent angle units |
| Tracking metric | Matches posterior position variance entries |
| Scanning metric interface | Valid synthetic normalization and missing-parameter handling |
| Allocation | Correct ordering and reversible conversions |
| Constraints | Valid and invalid synthetic configurations classified correctly |
| Objective | Correct scalar aggregation; no state mutation |
| Dataset | Correct tensor shapes and exact configuration/value pairing |
| GP | Finite posterior predictions; stable fitting |
| Acquisition | Valid candidate scoring and feasible selection |
| Joint BO | Improves known synthetic objective within documented budget |
| Alternating/coordinate BO | Fixed blocks preserved; complete observations stored |
| Experiment runner | Repeatable seeded runs; comparable budgets; outputs traceable to configuration |

Integration tests must cover configuration-to-simulator, evaluator-to-dataset, dataset-to-GP, acquisition-to-evaluator and results-to-analysis interfaces. Run small tests before full experiments. Do not suppress unexplained warnings. No research readiness claim until all applicable tests pass.

## 13. Milestones for the coding agent

- [x] M1: Establish repository, environment, dependency records, naming and units. *(2026-11-26: venv + pyproject + lock; 81 tests passing; see section 16.)*
- [x] M2: Implement nested Pydantic configuration and provenance register; preserve unresolved parameters. *(2026-11-26: 39-entry provenance register, 16 unresolved preserved as None; see section 16.)*
- [x] M3: Implement and validate standalone NumPy planar estimation baseline on synthetic observations. *(2026-11-26: Eqs. (1)-(15) implemented, synthetic validation in `results/m3_synthetic_validation/`; see section 16.)*
- [x] M4: Implement metric interfaces, generic allocations, feasibility and frozen-state evaluation. *(2026-11-26: scanning abstraction, allocation/constraints/accounting/objective evaluator, frozen snapshots, non-mutating candidate evaluation; 323 tests passing; see section 16.)*
- [x] M5: Build independently verifiable synthetic constrained objective and direct benchmark. *(2026-11-26: three synthetic benchmarks A/B/C with independently derived references — A analytic exact f\*=10.0, B enumeration exact f\*=11.0 with the analytic clamped inner optimum, C numerical approximate computed by an independent solver — plus independent reference solvers and a random-search comparator; see section 16.)*
- [x] M5 (duplicate line in this specification): same scope; the duplicate is marked once with the line above.
- [x] M6: Implement dataset, GP, acquisition, candidate selection and stopping criteria using BoTorch. *(2026-11-26: delivered INSIDE M5 as application-independent infrastructure per the working instructions — observation dataset, NumPy↔torch conversions, float64-CPU SingleTaskGP with explicit Matérn-5/2 + Normalize/Standardize, conventional LogExpectedImprovement (installed API verified: no xi), 500-candidate feasible pools, generic 15-step single-candidate loop with budget accounting and 7 stopping conditions; the README-M7 multi-mode scope (alternating/coordinate) remains NOT implemented.)*
- [ ] M7: Validate joint, alternating and coordinate-wise BO on synthetic objectives using equal budgets.
- [ ] M8: Implement reproducible experiment runner, logs, statistics and figures.
- [ ] M9: Document unresolved application-specific physical assumptions and scope of subsequent research; do not represent unfinished extensions as operational.

After each milestone, provide a concise summary of files created, tests executed, observed results, unresolved issues and deviations from this specification. Do not mark a milestone complete based only on code generation.

## 14. Open research decisions and limitations

- Reference SNR and numerical scanning/detector parameters are absent from the radar paper.
- Power and energy-budget choices are proposed extensions, not source-reported settings.
- Peak versus average power convention and waveform assumptions remain unresolved.
- Process-noise variance notation and zero-position EKF initialization require explicit treatment.
- Corridor-specific sensing and detection models require independent physical validation.
- The proposed hard per-slot constraint is stronger than the source's discounted constraint.
- A single-step objective differs from the source's long-term CMDP.
- A shared GP needs consistent input dimensions; dynamic population and contextual BO are future research questions.
- BO's benefit relative to direct optimization must be measured, especially for inexpensive objectives.
- No simulations, package installation or source-code validation have been performed as part of this specification.

## 15. Definition of done for the initial project

A reproducible research repository exists with validated configuration models, an independently tested planar estimation baseline, interchangeable metric and objective interfaces, generic constrained BoTorch optimization, joint and block-wise search on synthetic benchmarks, tests and traceable experiment outputs. Missing physical parameters and future application-specific extensions remain explicitly marked. The README explains exactly what has been implemented, what has been tested and what remains unresolved.

## 16. Implementation status and usage (updated per milestone)

This section records what is actually implemented and tested. It does not add requirements; sections 1–15 remain the specification.

### Status: M1–M5 complete — M6+ milestones beyond M5's delivered scope not started

| Milestone | State | Evidence |
|---|---|---|
| M1 repository/environment/dependencies | Done | `.venv` (Python 3.11.2), `pyproject.toml`, `requirements-lock.txt` (62 pinned deps), `.gitignore` |
| M2 config + provenance | Done | `radar_bo/config/` (7 models + register), `provenance_register.json` (62 entries, 17 unresolved preserved as `None`), layer-separation guard tests |
| M3 planar estimation baseline | Done | `radar_bo/environment/`, `radar_bo/radar/` (EKF parts), `radar_bo/numerics.py` — Eqs. (1)-(15); synthetic validation in `results/m3_synthetic_validation/` (regenerable) |
| M4 metric/allocation/feasibility/snapshot interfaces | Done | `radar_bo/radar/scanning_model.py` (Eq. (21)), `radar_bo/optimization/` (allocation, constraints, accounting, Eq. (23) evaluator), `radar_bo/simulation/` (frozen snapshots), `radar_bo/experiments/m4_integration_validation.py` |
| M5 generic BO engine (incl. README-M6 components) | Done | `radar_bo/optimization/bo/` (problem interface, benchmarks A/B/C, reference solvers, random search, dataset, conversions, GP surrogate, LogEI acquisition, candidate pools, results records, 15-step loop), `radar_bo/experiments/m5_synthetic_bo_validation.py`, `results/m5_synthetic_bo_validation/` (seeds 0-9); 469 tests passing |
| M7 joint/alternating/coordinate multi-mode validation, M8 experiment runner/figures, M9 final documentation | Not started | M5 implements the single-candidate JOINT mode only; alternating/coordinate search modes are not implemented |

### Installation

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # exact pinned environment
.venv/bin/pip install -e .                       # the radar_bo package itself
```

The lock was frozen from the working environment: numpy 2.4.6, pydantic 2.14.4, torch 2.15.0 (+cu132 wheel; CUDA libraries are bundled by the PyPI wheel but execution is CPU-only — `torch.cuda.is_available()` is False here), gpytorch 1.15.2, botorch 0.18.1, scipy 1.17.1, matplotlib 3.11.2, pandas 3.1.1, pytest 9.2.0.

### Test execution

```bash
.venv/bin/python -m pytest        # 469 passed (~4.5 min): M1-M5 config + estimation + interfaces + generic BO engine
```

### Experiment execution

```bash
# M3 synthetic validation (single target, seeded): config, provenance, metrics,
# trajectory data and a 2x2 diagnostic figure under results/m3_synthetic_validation/
.venv/bin/python -m radar_bo.experiments.m3_synthetic_validation

# M4 synthetic multi-task integration validation: feasibility classification under
# both budget semantics, Eq. (23) aggregation with hand-checked examples, non-mutation
# guarantees, order invariance; artifacts under results/m4_integration_validation/
.venv/bin/python -m radar_bo.experiments.m4_integration_validation

# M5 generic synthetic BO validation (benchmarks A/B/C x seeds 0-9, BO vs random
# search, budget 100, initial design 30; ~1 h wall-clock); artifacts under
# results/m5_synthetic_bo_validation/
.venv/bin/python -m radar_bo.experiments.m5_synthetic_bo_validation

# Regenerate the authoritative provenance register snapshot
.venv/bin/python -m radar_bo.config.provenance provenance_register.json
```

Every run saves its exact serialized `SimulationConfig` (`cfg.save_json(...)`) together with a provenance register. Experiments that require unresolved parameters fail with `UnresolvedParameterError` listing the missing names and units — values are never invented.

### Repository layout (maps the suggested structure in section 3)

```
radar_bo/                  package root (editable install; suggested top-level dirs are subpackages)
├── config/                tracking/scanning/resource/environment/optimization/experiment configs,
│                          SimulationConfig master model, provenance register   (M2)
├── environment/           TargetState/GaussianState/Observation representations, CV motion
│                          model (Lu et al. Eqs. (1)-(3))                        (M3)
├── radar/                 measurement model (Eqs. (4), (7)), SNR noise (Eqs. (5), (6), (8)),
│                          EKF (Eqs. (9)-(13)), initialization policies, tracking cost
│                          (Eqs. (14)-(15)), scanning-performance abstraction (Eq. (21)) (M3+M4)
├── optimization/          ResourceAllocation, ConstraintManager, ResourceAccounting,
│                          Eq. (23) objective evaluator, tracking-cost providers  (M4)
│   └── bo/                GENERIC BO engine, radar-free (M5): problem interface,
│                          benchmarks A/B/C, reference solvers, random search, dataset,
│                          conversions, GP surrogate, LogEI acquisition, candidate
│                          pools, result records, 15-step loop
├── simulation/            authoritative SimulationState + frozen estimator-facing snapshots (M4)
├── experiments/           m3_synthetic_validation.py, m4_integration_validation.py,
│                          m5_synthetic_bo_validation.py
├── numerics.py            shared validation helpers (shapes, finiteness, covariance, angles)
└── tests/                 469 tests incl. layer-separation and ground-truth-leakage guards
provenance_register.json   authoritative parameter register snapshot (regenerable)
requirements-lock.txt      pinned dependency lock
notes/paper_extracts/      plain-text extractions of the two reference PDFs (gitignored, regenerable)
```

Planned but not yet created: `analysis/`, `main.py` and the multi-mode search (alternating/coordinate) on top of the generic engine.

### M3+M4 component-to-equation mapping (Lu et al., ICC 2024)

| Component | Paper source | Notes |
|---|---|---|
| `environment/motion_model.py` | Eqs. (1)-(3) | F from Eq. (2) with revisit interval; Q from Eq. (3) multiplied DIRECTLY by the configured process covariance multiplier (25 under `table_variance`, 625 under `equation_literal`; M4 review terminology); stochastic propagation samples per-axis accelerations through a low-rank factor G with Q = G G^T q_a (exact: Q has rank 2) |
| `radar/measurement_model.py` | Eqs. (4), (7) | quadrant-aware atan2 (documented convention for the printed tan^-1(y/x)); Jacobian validated against numerical differentiation; singular origin raises `MeasurementGeometryError` |
| `radar/measurement_noise.py` | Eqs. (5), (6), (8) | Eq. (6) = reference variance DIVIDED by linear SNR (settled; the transient multiplication reading was a PDF-extraction artifact — see provenance `tracking.measurement_variance_semantics`); SNR0 unresolved -> `UnresolvedParameterError` |
| `radar/extended_kalman_filter.py` | Eqs. (9)-(13) | prior/posterior distinction preserved; gain via `linalg.solve` (no explicit inverse); angular residuals wrapped to [-pi, pi); optional Joseph-form update documented as numerically equivalent to Eq. (13) (tested); symmetrization only, no eigenvalue clipping |
| `radar/initialization.py` | p. 2786 + M3 review | `paper_zero_identity` diagnostic mode fails loudly at the origin; `measurement_driven` operational policy uses only the first observation plus configured velocity priors; step-0 observation is never reused as an update measurement (sequencing test) |
| `radar/tracking_cost.py` | Eqs. (14), (15) | standalone trace(E P E^T); validated on hand-constructed covariances |
| `radar/scanning_model.py` | Eq. (21) | area-ratio metric Gamma = S_max/S0, UNCAPPED; physical Eqs. (16)-(20) NOT implemented (parameters unspecified, Table I); substitution interface `ScanningPerformanceModel`; `SyntheticLinearScanningModel` clearly labeled synthetic |
| `optimization/constraints.py` | Eq. (22) | per-slot budget under BOTH semantics: proposed `hard_per_slot` (enforced, violations infeasible) vs paper-reference `discounted_cumulative` (slot term recorded as informational; horizon judgment out of M4 scope) — never treated as equivalent |
| `optimization/resource_accounting.py` | Eq. (24) | tau_s = T0*max(0, 1 - sum(tau)/T0) paper convention verbatim; raw deficit recorded separately so clipping never hides an exceedance; peak/average/duty-cycle power conventions separated |
| `optimization/objective.py` | Eq. (23) | U = -sum_n w_n c_n + beta*Gamma over explicit provider/model interfaces; beta = 1e6 paper value; uniform w_n is the documented proposed default; no GP/acquisition/BO code |

### M5 generic-BO-engine decision mapping (sources separated)

| Decision | Classification | Notes |
|---|---|---|
| GP surrogate: `SingleTaskGP` + Matérn-5/2 | **Benzaghta methodology** (methodological adaptation) | Paper: Matérn-5/2 GP. The EXPLICIT kernel configuration is a **library necessity**: the installed BoTorch 0.18.1 `SingleTaskGP` default is `RBFKernel` — verified at runtime and by a dedicated test; the kernel is never assumed |
| Acquisition: conventional `LogExpectedImprovement` WITHOUT xi | **Conventional BO + library decision** | Benzaghta reports EI with xi = 0.01; the INSTALLED `LogExpectedImprovement` accepts NO xi parameter (verified via `inspect.signature` + runtime guard + test). Conventional log-EI without an offset is the documented methodological adaptation; the paper value survives only in the provenance register (`optimization.acquisition_xi`) — no no-effect config field is retained |
| Input `Normalize(d)` / outcome `Standardize(1)` as BoTorch transforms | **Library decision** | ONE normalization mechanism (BoTorch `Normalize`); no manual pre-normalization; constraint checks and selection stay in ORIGINAL coordinates; `Standardize` is the outcome transform only (dataset keeps original units); posterior and `best_f` share the original-unit coordinate system |
| Observation noise INFERRED by the likelihood | **Library behavior, documented** | Deterministic benchmarks declare no noise, but the fitted likelihood noise is in general NOT numerically zero; latent predictive variance is kept distinct from observation-noise variance |
| Initial design: seeded uniform feasible random sampling, size 30 | **Proposed/experimental** | Not paper-derived; deliberately NOT Latin hypercube/Sobol; distinct from the acquisition search |
| Candidate pool 500 per iteration | **Benzaghta setting** (`Nc = 500`) | Pool scoring consumes ZERO objective evaluations; feasibility enforced by rejection (never repaired); duplicates rejected and counted |
| Evaluation budget (100 provisional in the M5 experiment) | **Synthetic/experimental** | No paper default exists; `evaluation_budget` stays UNRESOLVED in the register and is set per experiment (`ensure_ready` fails otherwise); initial evaluations, BO evaluations AND failed attempts all consume it |
| Refit per observation (fresh `fit_gpytorch_mll`, no warm-start) | **Proposed/experimental** | M5 initial policy; diagnostics recorded per fit (size, duration, status, warnings, noise, lengthscales) |
| Improvement tolerance / patience / max-iterations / runtime budget | **Proposed/experimental** | Stopping reason always recorded exactly; termination is NEVER described as proof of global convergence |
| Incumbent = best OBSERVED feasible configuration | **Proposed design rule** | Never replaced by an unevaluated GP prediction; history nondecreasing for deterministic problems |
| Duplicate policy (tolerance 1e-9, never silently merged) | **Proposed** | Deterministic benchmarks skip known configurations; noisy problems may repeat deliberately |
| Benchmarks A/B/C and their references | **Synthetic (self-defined)** | NOT from either paper; A: analytic exact f\*=10.0 at (3.5, 7.2); B: enumeration exact f\*=11.0 at (d=2, x=2) with the analytic clamped inner optimum; C: numerical approximate, value computed by the solver and recorded (never assumed) |
| Random-search comparator | **Proposed comparison method** | Same interface, same budget, same seed; records every evaluation; initial sampling cost counted. Comparability note: BO's initial design and random search share the SAME seeded uniform sampler, so with equal seeds RS's sequence begins with the BO initial design (exact BO/RS ties can result and are reported as measured) |

### Current limitations

- M3+M4 validate numerical and pipeline correctness only: all experiment artifacts are synthetic-validation records, NOT reproductions of the papers' experiments and NOT operational surveillance capability. The single-run M3 RMS position error is a numerical-validation result, NOT evidence of statistical filter consistency (no consistency analysis, e.g. NEES/NIS, has been performed).
- SNR0, all scanning physics parameters, the paper's S0, and task utility weights (uniform default) remain `None`/proposed where applicable; the synthetic validations supply explicitly labeled synthetic values to the components that need them, while the corresponding config fields stay unresolved.
- The synthetic scanning model is NOT validated radar physics; its outputs must never be presented as coverage, detectable range or detection probability results.
- The M4 tracking-cost provider is diagnostic and allocation-independent (paper Eq. (14) on current posteriors); a candidate's effect on FUTURE tracking requires simulation advancement (M5+).
- Dwell time is fixed at the paper reference tau0 = 1 s in the M3 validation; dwell allocation enters M4 only through the resource interfaces (no optimization).
- The `paper_zero_identity` initialization cannot run a full tracking episode from the origin (by design, diagnostic mode).
- M5 limitations: the generic engine implements the SINGLE-candidate JOINT mode only (no batching, no alternating/coordinate modes); acquisition supports conventional LogEI for low-noise/deterministic objectives (no noisy qLog variant is exercised); the GP likelihood noise is INFERRED (never forced to zero) even for deterministic benchmarks; the M5 validation budgets (100 evaluations, initial design 30) are provisional synthetic settings, not paper-derived; the seed 0-9 comparison is descriptive over 10 seeds and makes no superiority claim; BO results that fail to beat random search on some seeds are reported as measured; termination at a budget is not proof of global convergence; benchmark C's reference is numerical_approximate and may be exceeded legitimately.
- The PyPI torch wheel bundles CUDA runtime libraries (~4 GB installed) although only CPU execution is used; this is a wheel-availability artifact, not a GPU requirement.
- Ambiguity handling (switchable, documented defaults): process covariance multiplier (`table_variance` = 25 default — a reading of Table II units/prose, NOT an author-confirmed correction; `equation_literal` = 625), initialization policy (`paper_zero_identity` diagnostic default / `measurement_driven` operational / `experimental_custom`), covariance update (`paper_equation_13` default / `joseph`), time-budget semantics (`discounted_cumulative` paper default; `hard_per_slot` proposed BO profile — implemented in M4 and never treated as equivalent to the paper semantics).


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
- [x] M2: Implement nested Pydantic configuration and provenance register; preserve unresolved parameters. *(2026-11-26: 36-entry provenance register, 14 unresolved preserved as None; see section 16.)*
- [ ] M3: Implement and validate standalone NumPy planar estimation baseline on synthetic observations.
- [ ] M4: Implement metric interfaces, generic allocations, feasibility and frozen-state evaluation.
- [ ] M5: Build independently verifiable synthetic constrained objective and direct benchmark.
- [ ] M6: Implement dataset, GP, acquisition, candidate selection and stopping criteria using BoTorch.
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

### Status: M1 and M2 complete — M3+ not started

| Milestone | State | Evidence |
|---|---|---|
| M1 repository/environment/dependencies | Done | `.venv` (Python 3.11.2), `pyproject.toml`, `requirements-lock.txt` (62 pinned deps), `.gitignore` |
| M2 config + provenance | Done | `radar_bo/config/` (7 models + register), `provenance_register.json` (36 entries, 14 unresolved preserved as `None`), 81 tests passing |
| M3–M9 | Not started | No simulation, estimation, or optimization code exists yet |

### Installation

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # exact pinned environment
.venv/bin/pip install -e .                       # the radar_bo package itself
```

The lock was frozen from the working environment: numpy 2.4.6, pydantic 2.14.4, torch 2.15.0 (+cu132 wheel; CUDA libraries are bundled by the PyPI wheel but execution is CPU-only — `torch.cuda.is_available()` is False here), gpytorch 1.15.2, botorch 0.18.1, scipy 1.17.1, matplotlib 3.11.2, pandas 3.1.1, pytest 9.2.0.

### Test execution

```bash
.venv/bin/python -m pytest        # 81 passed (0.3 s)
```

### Experiment execution

No experiments exist yet (M3+). Currently runnable utilities:

```bash
.venv/bin/python -m radar_bo.config.provenance provenance_register.json   # regenerate register
```

Every future run must save its exact serialized `SimulationConfig` (`cfg.save_json(...)`) together with its provenance register. Experiments that require unresolved parameters fail with `UnresolvedParameterError` listing the missing names and units — values are never invented.

### Repository layout (maps the suggested structure in section 3)

```
radar_bo/                  package root (editable install; suggested top-level dirs are subpackages)
├── config/                tracking/scanning/resource/environment/optimization/experiment configs,
│                          SimulationConfig master model, provenance register   (M2)
└── tests/                 81 unit tests incl. a layer-separation guard test   (M1+M2)
provenance_register.json   authoritative parameter register snapshot (regenerable)
requirements-lock.txt      pinned dependency lock
notes/paper_extracts/      plain-text extractions of the two reference PDFs (gitignored, regenerable)
```

Planned but not yet created: `environment/`, `radar/`, `optimization/`, `benchmarks/`, `simulation/`, `experiments/`, `analysis/`, `main.py`.

### Current limitations

- No EKF, simulation, BO, or experiment code exists; nothing here validates the papers' numerical results.
- All scanning physics parameters and SNR0 are unresolved (`None` + provenance records); any experiment needing them fails clearly.
- The PyPI torch wheel bundles CUDA runtime libraries (~4 GB installed) although only CPU execution is used; this is a wheel-availability artifact, not a GPU requirement.
- Ambiguity handling (switchable, documented defaults): Eq. (3) σ_w interpretation (`table_variance` default / `equation_literal`), zero-vector initialization (`paper_zero_identity` default / `proposed_measurement_driven`), Eq. (6) variance-vs-SNR direction (`paper_as_printed` default / `proposed_inverse_snr`), time-budget semantics (`discounted_cumulative` default / `hard_per_slot`).


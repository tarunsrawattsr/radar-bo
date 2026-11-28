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
- [x] M6: Implement dataset, GP, acquisition, candidate selection and stopping criteria using BoTorch. *(2026-11-26: delivered INSIDE M5 as application-independent infrastructure per the working instructions — observation dataset, NumPy↔torch conversions, float64-CPU SingleTaskGP with explicit Matérn-5/2 + Normalize/Standardize, conventional LogExpectedImprovement (installed API verified: no xi), 500-candidate feasible pools, generic 15-step single-candidate loop with budget accounting and 7 stopping conditions.)*
- [x] M7: Validate joint, alternating and coordinate-wise BO on synthetic objectives using equal budgets. *(2026-11-26: delivered as instruction-stream M6 — ONE generic block-cyclic engine with interchangeable variable-block structures (joint = single full block, the M5 baseline; alternating = two user-defined blocks; pair-wise = two-variable blocks), shared dataset/surrogate, improvement-only current-configuration acceptance, global-budget accounting, 4/8/16-D synthetic benchmarks with independent references, 10-seed comparison vs random search; equal total budgets across architectures. Alternating/coordinate are NOT reproduced radar algorithms.)*
- [x] M8A: Scientific Model Review and Parameter Resolution. *(2026-11-26: delivered as instruction-stream item — final M7 regression re-verified (668 passed on `371952a`); run-count reconciliation (220 archived records exactly; "230" was a narrative arithmetic error, corrected); validated immutable shared-initial-design interface (engine + framework `explicit_shared`, backward compatible, 11 new tests); full provenance audit (81 entries / 17 unresolved, resolution paths per unresolved parameter, NO invented values); literature review incl. the authors' extended arXiv version (2606.05526) with version-difference table; consolidated mathematical specification with dimensional findings; assumptions/limitations, application-validation plan, research-gap assessment; seven documents under `docs/`. STOPPED before M8B — M8B requires explicit approval of the scientific assumptions and proposed model specifications in `docs/`.)*
- [ ] M8: Implement reproducible experiment runner, logs, statistics and figures. *(figures/statistics exist per experiment under results/ and tracked archives under research_results/; a unified runner/analysis package remains open)*
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

### Status: M1–M7 (instruction stream) + M8A (scientific model review) complete — M8B requires explicit approval; M8/M9 of the roadmap NOT started

| Milestone | State | Evidence |
|---|---|---|
| M1 repository/environment/dependencies | Done | `.venv` (Python 3.11.2), `pyproject.toml`, `requirements-lock.txt` (62 pinned deps), `.gitignore` |
| M2 config + provenance | Done | `radar_bo/config/` (7 models + register), `provenance_register.json` (81 entries after the two M8A additions, 17 unresolved preserved as `None`), layer-separation guard tests |
| M3 planar estimation baseline | Done | `radar_bo/environment/`, `radar_bo/radar/` (EKF parts), `radar_bo/numerics.py` — Eqs. (1)-(15); synthetic validation in `results/m3_synthetic_validation/` (regenerable) |
| M4 metric/allocation/feasibility/snapshot interfaces | Done | `radar_bo/radar/scanning_model.py` (Eq. (21)), `radar_bo/optimization/` (allocation, constraints, accounting, Eq. (23) evaluator), `radar_bo/simulation/` (frozen snapshots), `radar_bo/experiments/m4_integration_validation.py` |
| M5 generic BO engine (incl. README-M6 components) | Done | `radar_bo/optimization/bo/` (problem interface, benchmarks A/B/C, reference solvers, random search, dataset, conversions, GP surrogate, LogEI acquisition, candidate pools, results records, 15-step loop), `radar_bo/experiments/m5_synthetic_bo_validation.py`, `results/m5_synthetic_bo_validation/` (seeds 0-9; durably archived in `research_results/`) |
| M6 block architectures (README-M7 scope) | Done | `radar_bo/optimization/bo/blocks.py`, `scheduling.py`, `block_candidates.py`, `benchmarks_highdim.py` (4/8/16-D), block-cyclic refactor of `bo_loop.py` (joint preserved), `radar_bo/experiments/m6_block_architecture_comparison.py`, `m6_generate_figures.py`, `m6_merge_results.py`; pilot + full matrix executed (220 archived records exactly; the earlier narrative said "230 runs" — arithmetic/reporting error corrected in M8A via `experiment.m8a_run_count_reconciliation`, archives always internally consistent), measured narrative in "End-of-M6 status"; durable archive `research_results/m6_block_architecture_full_evidence.json`; M7 review corrections in "M7 targeted M6 review" |
| M7 experimental framework (instruction stream) | Done | `radar_bo/experiments/framework/` (config, benchmark/algorithm registries, matrix, checkpointed runner, metrics, runtime analysis, candidate profiling, archive integrity, sensitivity, reporting, figures, simulator adapter), validation experiment + sensitivity pilot, review corrections + archive validation + paired summaries (derived artifacts in `research_results/`); see "M7 experimental framework" below |
| M8A scientific model review (instruction stream) | Done | shared-initial-design interface (`optimization_config.py` + `bo_loop.py` + framework `explicit_shared`, 11 tests in `radar_bo/tests/test_shared_initial_design.py`), provenance 81/17 + resolution paths + extended-version cross-reference (`radar_bo/experiments/m8a_generate_parameter_register.py` → `docs/PARAMETER_RESOLUTION_REGISTER.md`), seven review documents under `docs/`; see "M8A scientific model review" below |
| M8B (instruction stream: application optimization) | Not started | REQUIRES explicit user approval of scientific assumptions + proposed model specifications (`docs/M8A_SCIENTIFIC_MODEL_REVIEW.md` §10, `docs/APPLICATION_VALIDATION_PLAN.md`) |
| M8/M9 (roadmap: runner package / final docs) | Not started | Require explicit approval |

### Installation

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # exact pinned environment
.venv/bin/pip install -e .                       # the radar_bo package itself
```

The lock was frozen from the working environment: numpy 2.4.6, pydantic 2.14.4, torch 2.15.0 (+cu132 wheel; CUDA libraries are bundled by the PyPI wheel but execution is CPU-only — `torch.cuda.is_available()` is False here), gpytorch 1.15.2, botorch 0.18.1, scipy 1.17.1, matplotlib 3.11.2, pandas 3.1.1, pytest 9.2.0.

### Test execution

```bash
.venv/bin/python -m pytest        # 680 passed (~19 min): M1-M7 config + estimation + interfaces + generic BO engine + block architectures + experimental framework + M8A shared-initial-design suite
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
# search, budget 100, initial design 30; ~30 min wall-clock); artifacts under
# results/m5_synthetic_bo_validation/
.venv/bin/python -m radar_bo.experiments.m5_synthetic_bo_validation

# M6 block-architecture comparison: pilot first (validation checks + cost estimate),
# then the full matrix (4 primary benchmarks x {RS, joint, alternating, pairwise}
# x seeds 0-9 + separable 8d/16d dimension study; ~2 h wall-clock on 2 cores);
# artifacts under results/m6_primary/, results/m6_dimension_study/,
# results/m6_block_architecture_comparison/ (merged) and results/m6_pilot/
.venv/bin/python -m radar_bo.experiments.m6_block_architecture_comparison results/m6_pilot --pilot
.venv/bin/python -m radar_bo.experiments.m6_block_architecture_comparison results/m6_primary --benchmarks=separable_4d,coupled_8d,constrained_mixed_8d,block_stagnation_4d
.venv/bin/python -m radar_bo.experiments.m6_block_architecture_comparison results/m6_dimension_study --benchmarks=separable_8d,separable_16d --methods=random_search,joint,pairwise
.venv/bin/python -m radar_bo.experiments.m6_merge_results    # assemble merged results + durable archive

# M7 experimental framework (validation matrix, sensitivity pilot, review + analyses)
.venv/bin/python -m radar_bo.experiments.m7_validation_experiment    # ~11 min; validates runner/checkpoint/metrics/reporting
.venv/bin/python -m radar_bo.experiments.m7_sensitivity_pilot       # explicit 3-scenario sensitivity pilot
.venv/bin/python -m radar_bo.experiments.m7_m6_review               # targeted M6 review + archive validation (derived artifacts)
.venv/bin/python -m radar_bo.experiments.m7_paired_summary          # paired M5/M6 statistical summaries (derived)
.venv/bin/python -m radar_bo.experiments.m7_runtime_profiling       # runtime attribution profiling (derived artifact)

# M6 figures (labeled generic synthetic M6 experiment; regenerable from tracked data)
.venv/bin/python -m radar_bo.experiments.m6_generate_figures results/m6_block_architecture_comparison

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
│   └── bo/                GENERIC BO engine, radar-free (M5+M6): problem interface,
│                          benchmarks A/B/C + high-dim 4/8/16-D, reference solvers,
│                          random search, dataset, conversions, GP surrogate, LogEI
│                          acquisition, candidate pools (full-space + block-restricted),
│                          variable blocks + cyclic scheduler, result records, loop
├── simulation/            authoritative SimulationState + frozen estimator-facing snapshots (M4)
├── experiments/           m3_synthetic_validation.py, m4_integration_validation.py,
│                          m5_synthetic_bo_validation.py, m6_block_architecture_comparison.py,
│                          m6_generate_figures.py, m6_merge_results.py,
│                          m7_runtime_profiling.py, m7_m6_review.py, m7_paired_summary.py,
│                          m7_sensitivity_pilot.py, m7_validation_experiment.py,
│                          m7_rebuild_m6_results.py
│   └── framework/         M7 experimental framework: unified experiment config,
│                           benchmark/algorithm registries, matrix generation,
│                           checkpointed runner with resume, gap/closure/threshold
│                           metrics, paired statistics (bootstrap + Wilcoxon/Holm),
│                           runtime decomposition, candidate profiling, archive
│                           integrity + checksums, sensitivity scenarios, report
│                           + figure generation, simulator-interface adapter
├── numerics.py            shared validation helpers (shapes, finiteness, covariance, angles)
└── tests/                 680 tests incl. layer-separation, ground-truth-leakage and shared-design guards

docs/                      M8A review deliverables (7 documents; see "M8A scientific model review")
provenance_register.json   authoritative parameter register snapshot (regenerable)
requirements-lock.txt      pinned dependency lock
research_results/          TRACKED durable evidence archives (per-seed histories, references,
                           runtimes, versions, commit ids; see research_results/README.md)
notes/paper_extracts/      plain-text extractions of the two reference PDFs (gitignored);
                           LOST in the 2026-11-26 sandbox reset and not regenerated —
                           literature facts re-verified from arXiv extended versions (see docs/)
```

Planned but not yet created: `analysis/` and `main.py` (unified experiment-runner package).

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

### M6 block-architecture design mapping (sources separated; instruction 30)

| Topic | Classification | Notes |
|---|---|---|
| Variable-block representation (`optimization/bo/blocks.py`) | **Proposed M6 design** | Index-based, application-free; blocks must not overlap; uncovered variables stay fixed at the current configuration and are REPORTED; block metadata is evaluation metadata, never GP input features |
| Deterministic cyclic scheduling (`scheduling.py`) | **Proposed M6 design** | One complete cycle = every block visited or skipped once; visit tickets record block id, cycle number, within-cycle index; strategies (joint/alternating/pair-wise) are block CONFIGURATIONS of one engine, not separate implementations (no redundant joint copy — the M5 loop IS the single-block configuration) |
| Alternating BO | **Proposed architecture** (block-wise analogue of the Benzaghta per-BS scheme, methodological adaptation) | Two user-defined blocks; per visit: block candidates around the current configuration → shared-GP LogEI scoring → one evaluation → acceptance/incumbent updates; NOT a reproduced radar algorithm |
| Pair-wise coordinate BO | **Proposed architecture** | Multiple two-variable blocks via the same engine; the declared `search_mode` resolves the structural ambiguity when two two-variable blocks match both readings |
| Current configuration vs global incumbent | **Proposed design rule** | Two distinct objects; the current configuration is the block-search center (improvement-only acceptance, a PROPOSED policy), the incumbent is the best OBSERVED feasible configuration of the whole run; both always refer to stored observations; never replaced by unevaluated GP predictions |
| Block candidate generation (`block_candidates.py`) | **Proposed design rule** | Only active coordinates sampled; fixed coordinates preserved EXACTLY; every COMPLETE candidate validated (shared constraints may span blocks); infeasible candidates rejected, never repaired; empty block pools REPORTED and skipped — an empty individual block does not terminate the run while other blocks remain searchable |
| Shared dataset + shared GP | **Proposed architecture** (single-GP default) | M×d full-dimension training tensors regardless of block size; one `SingleTaskGP` (Matérn-5/2, Normalize/Standardize, float64 CPU, inferred noise) for all blocks; separate per-block GPs would be a distinct unapproved method; reducing the block dimension does NOT reduce GP training cost |
| Within-block optimization | **Proposed/experimental** | 1 evaluation per block visit by default (`evaluations_per_block_visit`, configurable); one acquisition-selected evaluation is NOT an exact block maximization; multiple evaluations share the global budget |
| Block-wise stagnation | **Demonstrated limitation** | `BlockStagnationLP` (analytic LP): from budget-tight configurations the alternating a/b grouping provably cannot improve while the joint optimum exists; skipped-block and no-improvement-cycle counters are recorded; stagnation never proves global optimality |
| M6 synthetic benchmarks 4/8/16-D | **Synthetic (self-defined)** | Separable quadratic (analytic exact f\*=10d), coupled quadratic via fixed PD matrix (analytic exact f\*=20), constrained mixed 8-D (numerical_approximate: enumeration + KKT multiplier bisection — a BO result exceeding it is reported and investigated, never clipped), block-stagnation LP (analytic exact f\*=9.6) |
| M6 experiment settings | **Synthetic/experimental** | Budget 100 total ATTEMPTED evaluations shared by all architectures (not per block), initial design 30, pool 500, seeds 0-9, identical initial designs per seed across architectures; pilot phase validated correctness before the full matrix |

### M6 unit-test coverage audit (instruction-stream item 27; 20 areas)

| # | Requirement area | Tests |
|---|---|---|
| 1 | Block representation: unique id, ordered active indices, block dimensionality, names/types; index-range + intra-block-duplicate + partition-coverage validation | `test_blocks.py` (21 tests: `test_valid_block_resolves`, `test_index_outside_dimension_rejected`, `test_duplicate_indices_within_block_rejected`, `test_empty_block_rejected`, `test_overlap_rejected`, `test_duplicate_block_ids_rejected`, `test_full_coverage_requirement`, `test_uncovered_indices_reported`, …) |
| 2 | Radar-free, application-free blocks (index-based on any problem) | `test_blocks.py::test_blocks_on_mixed_variable_problem`; layer-separation guards |
| 3 | Scheduling interface + deterministic cyclic scheduler; tickets carry block id, cycle number, within-cycle index | `test_blocks.py::test_cyclic_order_and_ticket_fields`, `test_strategy_constructors`; `test_block_bo.py::test_partial_cycle_termination` |
| 4 | ONE generic engine; M5 joint behavior preserved exactly (no second joint implementation) | `test_block_bo.py::test_joint_mode_matches_m5_behavior`, `::test_joint_trajectory_bit_for_bit_m5` (frozen M5 trajectory, bit-for-bit) |
| 5 | Mode compatibility + declared-mode resolution (1 block rejected; ≥2 blocks; `coordinate` iff every block has exactly 2 variables; ambiguity resolved by `search_mode`) | `test_block_bo.py::test_two_single_variable_blocks_declared_alternating`, `::test_two_two_variable_blocks_declared_coordinate`, `::test_search_mode_must_match_blocks`; `test_blocks.py::test_pairwise_requires_two_variables_per_block`, `::test_pairwise_requires_at_least_two_blocks`, `::test_alternating_requires_nonempty_groups` |
| 6 | Current configuration ≠ global incumbent; both from stored observations, never GP predictions; never inconsistent with fixed coordinates | `test_block_bo.py::test_current_config_and_incumbent_distinct`, `::test_incumbent_always_evaluated_never_predicted`, `::test_fixed_coordinates_traceable_across_block_visits` |
| 7 | Block candidates vary only active coordinates; fixed coordinates preserved exactly; COMPLETE-vector validation against shared constraints; invalid/duplicates rejected, never repaired | `test_block_candidates.py` (9 tests: `test_fixed_coordinates_preserved_exactly`, `test_shared_constraint_validated_on_complete_vectors`, `test_all_rejected_returns_empty_pool_with_report`, `test_current_config_excluded_as_duplicate`, `test_evaluated_configs_excluded`, …) |
| 8 | Empty-pool handling: skip + recorded reason, run continues; all-blocks-exhausted cycle terminates with exact reason | `test_block_bo.py::test_empty_block_skipped_not_terminal`, `::test_all_blocks_skipped_stops_with_exact_reason`; `test_m6_integration.py::test_empty_blocks_never_create_infinite_loops` |
| 9 | Block-restricted stagnation demonstrable while a better joint solution exists | `test_benchmarks_highdim.py::test_no_single_block_improvement_at_tight_config`, `::test_joint_improvement_exists`; `test_block_bo.py::test_alternating_stagnation_on_lp_benchmark`; `test_m6_integration.py::test_pairwise_on_stagnation_lp` |
| 10 | Shared dataset holds complete M×d vectors; block ids metadata only; failures never enter GP training | `test_block_bo.py::test_dataset_stays_full_dimension_and_shared`; `test_m6_integration.py::test_failures_never_enter_gp_training` |
| 11 | ONE shared GP (SingleTaskGP, explicit Matérn-5/2, Normalize/Standardize, float64 CPU, M5 noise convention); block dimension does NOT shrink GP training dimension | `test_bo_loop.py` (M5 GP construction/convention tests) + full-dimension dataset assertions in `test_block_bo.py` |
| 12 | Block-restricted acquisition scoring on complete vectors; batched LogEI reused; score↔config association; scored candidates ≠ evaluations | `test_block_candidates.py::test_pool_size_is_not_evaluation_count`, `::test_reproducible_pool_for_same_seed`; `test_block_bo.py::test_budget_enforced_exactly`; `test_accounting_identities.py` |
| 13 | 1 evaluation per block visit (configurable); one selected evaluation ≠ exact block maximization | `test_block_bo.py::test_evaluations_per_block_visit_two`; visit accounting (`n_block_visits`) |
| 14 | Improvement-only current-config acceptance; failed-to-improve evaluations stay in the dataset | `test_block_bo.py::test_non_improving_observations_retained`, `::test_current_config_and_incumbent_distinct` |
| 15 | Alternating workflow (2 blocks) through the same engine | `test_block_bo.py::test_alternating_on_benchmark_a`; `test_m6_integration.py::test_complete_workflow_on_mixed_problem` |
| 16 | Pair-wise workflow (2-variable blocks) through the same engine; differences only in block definitions/scheduling | `test_m6_integration.py::test_pairwise_on_constrained_mixed_8d`, `::test_identical_initial_design_across_architectures`, `::test_unfavorable_comparison_is_valid_output` |
| 17 | Stopping distinguishes evaluations/iterations/visits/cycles/skips; fixed total budget; partial-cycle termination without overrun; exact reason recorded | `test_block_bo.py::test_budget_enforced_exactly`, `::test_partial_cycle_termination`, `::test_cycle_patience_stops`, `::test_max_iterations_still_applies`; `test_accounting_identities.py` (I1–I4 incl. injected-failure runs) |
| 18 | Global budget (never per block); requested/feasible/scored candidate counts reported | `test_accounting_identities.py::test_random_search_identities`; pool accounting fields in every run record |
| 19 | Reproducibility (same seed → same trajectory; serialization round trip with block fields) | `test_block_bo.py::test_reproducible_same_seed`, `::test_serialization_round_trip_with_block_fields` |
| 20 | Failure accounting (failures consume budget, carry block metadata, never enter the success dataset) | `test_block_bo.py::test_failures_consume_budget_and_carry_block_metadata`; `test_accounting_identities.py::test_joint_run_with_failures`, `::test_block_run_with_failures_and_k2`, `::test_all_failing_problem` |

### End-of-M6 status (measured results, honest reporting; instruction 22/23/32)

**Execution (phase 22):** pilot passed first (budget 40, initial 10, seeds 0–1; every per-run
validity check green, exit 0), then the full matrix ran as two parallel processes
(4 primary benchmarks × {RS, joint, alternating, pairwise} × seeds 0–9 + separable 8d/16d
dimension study × {RS, joint, pairwise} × seeds 0–9; budget 100 total, initial design 30,
pool 500, 1 evaluation per block visit — inherited M5 settings, NOT proven optimal).
Every one of the 230 BO/RS runs passed the in-run validity assertions (budget identity
`n_initial+n_bo == 100`, all observations feasible, incumbent ⊆ evaluated observations,
`BO observations + skipped == block visits`, per-evaluation incumbent-history resolution);
both processes exited 0. The two partials were merged with
`m6_merge_results.py` (per-benchmark blocks unchanged) and figures regenerated.
Merged results: `results/m6_block_architecture_comparison/` (gitignored, regenerable);
durable per-run evidence (including iteration diagnostics, skipped-block records, pool
accounting, out-of-block evidence, references, versions, commit id):
`research_results/m6_block_architecture_full_evidence.json` (tracked; ~9.9 MB JSON, no binaries).

**Primary benchmarks** (best-observed mean ± stdev over seeds 0–9; simple regret vs the
independent reference; objective-quality thresholds were FROZEN during the pilot phase,
before any full-matrix comparative result was examined — threshold = fraction × f\*,
0.90 everywhere except 0.95 for constrained_mixed):

| Benchmark (f\*) | random search | joint BO | alternating BO | pair-wise BO |
|---|---|---|---|---|
| separable_4d (40.0) | 34.807 ± 2.281 | 38.747 ± 0.618 | **39.906 ± 0.111** | **39.906 ± 0.111** |
| coupled_8d (20.0) | −11.239 ± 10.700 | −2.650 ± 6.012 | 14.888 ± 2.308 | **19.649 ± 0.196** |
| constrained_mixed_8d (39.53, numerical_approximate) | 38.486 ± 0.306 | 38.898 ± 0.226 | 38.858 ± 0.378 | **39.113 ± 0.292** |
| block_stagnation_4d (9.6) | 8.120 ± 0.443 | 8.670 ± 0.241 | 8.249 ± 0.911 | **8.984 ± 0.299** |

**Dimension study** (separable quadratic family, same budget):

| Dimension | random search | joint BO | pair-wise BO |
|---|---|---|---|
| separable_8d (f\* = 80) | 62.293 ± 5.653 | 66.935 ± 3.261 | **79.782 ± 0.128** |
| separable_16d (f\* = 160) | 81.238 ± 16.075 | 98.800 ± 8.860 | **157.796 ± 2.700** |

**Key findings (descriptive over 10 seeds; no superiority claim):**

1. **Architecture-structure match dominates in higher dimensions.** On the separable
   family, pair-wise decomposition reached the 90 % threshold in 10/10 seeds (8d: mean
   79.78, regret 0.22; 16d: mean 157.80, regret 2.20) while joint BO reached it in 1/10
   (8d) and 0/10 (16d) seeds (16d regret 61.2). Same seed → pairwise better than joint on
   10/10 seeds at both dimensions.
2. **Coupled 8d: block-restricted search around the moving current configuration beat
   joint LogEI under this budget** (pairwise regret 0.35, alternating 5.11, joint 22.65;
   10/10 seeds pairwise/alternating > joint). Random search is catastrophic here
   (regret 31.2) — GP guidance matters even when block-restricted.
3. **Constrained mixed 8d: all methods within ~0.6 regret** (best observation 39.4605 by
   pairwise, seed 7 — below the 39.53 numerical reference; **no `exceeds_numerical_reference`
   flag in any run**). Pairwise vs joint same-seed: 9 better / 1 worse. The 95 % threshold
   was already cleared by the shared initial design in most runs, so
   evaluations-to-threshold is not discriminative on this benchmark.
4. **Instruction-9 empty-pool behavior measured in the wild:** alternating on
   constrained_mixed_8d skipped a mean of 50.2 visits per run (discrete 2-variable block:
   4² = 16 distinct combinations get exhausted → `empty_feasible_pool_for_block` → skip,
   run continues); pairwise skipped 9.7/run. Skips are recorded per visit with reasons in
   the durable archive.
5. **Block-stagnation evidence:** the alternating grouping on the stagnation LP averages
   8.249 (vs joint 8.670, pairwise 8.984) — the provably-blocked grouping performs worst,
   and per-run stagnation counters (cycles without improvement, visits without accepted
   update, skipped blocks) are recorded. Stagnation is NOT a convergence proof.
6. **Out-of-block improvement events: 0 in all 230 runs.** By construction, block
   candidates vary only active coordinates and the current configuration carries fixed
   coordinates; the metric confirms clean improvement attribution (every incumbent
   improvement is attributable to the visited block — or the initial design).
7. **Sample vs computational efficiency are distinct (instruction 24, separate stages).**
   Joint BO wall-clock is GP-fit dominated (constrained_mixed: 37.1 s GP of 115.2 s total);
   block-restricted search on the same problem is candidate-GENERATION dominated
   (alternating: 251.5 s generation + 67.4 s feasibility validation of 324.7 s — small
   blocks sample mostly-infeasible corners and never repair; ~14–26 infeasible rejections
   per requested candidate slot). GP training dimension is M×d for every architecture:
   smaller blocks do NOT reduce GP fit cost (block runs even fit slightly more often).
8. **evaluations-to-threshold** (BO axis: 1 = initial design + k BO evaluations; RS axis:
   attempted evaluations; thresholds frozen pre-results): separable_4d joint 2.1 /
   alternating 2.5 / pairwise 2.5 (all 10/10 reached; RS 84.5, 2/10); coupled_8d pairwise
   21.4 (10/10) vs alternating 23.0 (1/10), joint never (0/10); separable_8d pairwise 7.9
   (10/10) vs joint 2.0 (1/10); separable_16d pairwise 41.9 (10/10) vs joint never.
9. All 140 BO runs on the primary benchmarks stopped at `evaluation_budget_exhausted`
   (fixed total budget respected; no early stopping); the joint-vs-random-search same-seed
   counts (e.g. separable_4d: joint better on 10/10) and all per-seed histories are in the
   merged results and the durable archive.

### M7 targeted M6 review (instruction-stream item 3; corrections, sources separated)

All corrections below are stored as DERIVED analysis artifacts
(`research_results/m7_m6_review_corrections.json`, `m7_m6_review_profiling.json`);
historical M5/M6 archive values are preserved unchanged.

| Item | Finding and correction |
|---|---|
| 3.1 runtime attribution | **RETRACTED:** "GP fitting dominates M5 runtime". Profiling on the current instrumented engine (9 runs, joint mode): GP fit ~13.6–17.0 s of 69.5–74.9 s walls (~20–23%); candidate generation ~54.9–58.5 s (~75–80%) is the largest component; acquisition ~0.7 s; objective ~5 ms. The M5-era report had ~44 s/run UNINSTRUMENTED; the M6 instrumentation plus this profiling pass identified it as candidate generation + nested feasibility. Profiling also EXPOSED a timer defect: the archived generation timer included nested feasibility validation (component sums slightly exceeded wall-clock) — fixed for M7+ records (disjoint boundaries); historical values kept, corrected values only in derived artifacts. No conclusive per-component attribution beyond the instrumented stages is claimed. |
| 3.2 candidate-generation 251.5 s | Traced: it is the MEAN across the 10 archived seeds (field `benchmarks.constrained_mixed_8d.aggregate.per_method.alternating.candidate_generation_runtime_mean_s` = 251.504). Per-run values 170.8–318.3 s (1.9x spread). Includes nested feasibility 50.9–81.9 s (mean 67.4); corrected derived mean excluding feasibility: 184.1 s. |
| 3.3 missing combinations | Alternating BO was deliberately OMITTED from the separable 8d/16d dimension study (design decision: the primary set covers alternating on separable_4d; three-method study halved wall-clock on 2 cores). Recorded as an explicit omission; no values substituted. No failed or incomplete runs occurred in the final M6 execution. |
| 3.4 block stagnation | The LP construction is a PROOF about the mechanism (alternating grouping provably cannot improve from budget-tight configurations); the measured M6 means (alternating worst: 8.249 vs joint 8.670, pairwise 8.984) are consistent empirical observations, NOT proofs. The out-of-block event detector (0 events in 230 runs) counts incumbent-improving observations differing from the previous incumbent OUTSIDE the visited block — because block candidates inherit fixed coordinates from the current configuration, near-zero events are largely a tautology of the block mechanism and must NOT be read as evidence that better joint configurations do not exist. |
| 3.5 chronology | Pilot (exit 0) → full-matrix first execution, deliberately stopped after ~25 min due to a recording deficiency (missing diagnostics/thresholds/out-of-block evidence) → recording upgrade → final execution (230 runs, exit 0). This was a DETERMINISTIC RE-EXECUTION of the same seeds/configs (identical random-search values verified), NOT an independent replication; no replication-stability claims are made. |
| 3.6 archive validation | Independent checks (item 4): M5 archive 30 records VALID (budget identity, monotone histories, stops, paired RS rows). M6 archive 220 records VALID (unique combos, budget identity re-derived from recorded config, feasibility recomputed via registry problems, incumbent correspondence, monotone BO histories, references consistent, metadata present) with TWO documented findings: (a) the nested-timer convention (see 3.1); (b) archived RANDOM-SEARCH rows store the full per-evaluation OBJECTIVE SEQUENCE under the key `incumbent_history` (not a best-so-far curve) — data intact (best == max(sequence) verified in all 60 rows), label wrong; corrected semantics live in the M7 runner records. |

### M7 experimental framework (instruction-stream M7)

**Architecture** (`radar_bo/experiments/framework/`, layer separation item 32 — the
experiment layer imports only config + generic BO interfaces + analysis libraries;
core layers never import experiment orchestration, guarded by tests):

| Module | Purpose (instruction item) |
|---|---|
| `experiment_config.py` | Pydantic `ExperimentConfig`: experiment id/description/labels, benchmarks, algorithms, dimensions, seeds, initial design, budgets, pool, GP, acquisition, stopping, runtime measurement, statistical settings, output dir, checkpoint policy (item 5); fingerprint = sha256 of canonical JSON |
| `benchmark_registry.py` | Registry: id, versioned implementation reference, math definition, variable names/types/bounds, constraints, direction, noise, independent reference + verification method; runner accesses benchmarks ONLY through the registry (item 6) |
| `algorithm_registry.py` | random_search / joint_bo / alternating_bo / pairwise_bo / reference — all REUSE the existing M5/M6 engine (no duplication); per-run architecture + version recorded (item 7) |
| `matrix.py` | Run = (benchmark, dimension, algorithm, seed, hyperparams); stable 16-hex run ids (sha256 over canonical identity); duplicate detection; incompatible combos validated BEFORE execution and listed as skipped with reasons; explicit manifest (item 8) |
| `checkpointing.py` | Atomic per-run JSON writes (tmp + `os.replace`); manifest statuses pending/running/completed/failed/skipped + reasons; resume with compatibility checks (config fingerprint, benchmark/algorithm versions, dependency versions, seeds, run ids); corrupted/incompatible records REJECTED explicitly (items 11-12) |
| `runner.py` | Loads config, resolves registries, fresh problem instances per run, executes, collects item-13 record structure (initial design, observations, failures, histories, counts, block/cycle + candidate diagnostics, runtimes, versions, commit, config id), verifies shared initialization FROM STORED RECORDS (item 10), writes summary. NO GP/acquisition logic (item 9) |
| `metrics.py` | Direction-aware objective gaps (exact → simple_regret; numerical → reference_gap; negatives retained, never clipped); normalized gap closure with explicit zero/negative-denominator handling; threshold analysis (50/90/99% gap closure, frozen pre-run; unreached reported explicitly); descriptive stats (n/mean/median/std/IQR); paired comparisons (per-seed diffs, W/L/T at tol 1e-9, d_z); bootstrap CIs resampling seed-level pairs (B=10000, seed 2026); Wilcoxon (documented H0/zero-handling/ties) + Holm family correction; convergence axes (attempted/successful/wall/cycles — attempted+wall exact for M7 records, unavailable for pre-M7, cycles never a primary cross-architecture axis); runtime decomposition (disjoint timers + explicit unattributed remainder) (items 15-23) |
| `runtime_analysis.py` | Per-method runtime aggregation with documented timer boundaries and the no-exact-sum disclaimer (item 23) |
| `candidate_profiling.py` | Pool-report analysis: rejection/duplicate rates, skips, time per delivered candidate; PROVEN finite-space exhaustion vs sampling failure classification; M6 algorithm preserved as versioned baseline (item 24) |
| `archive_integrity.py` | sha256 checksums (`ARCHIVE_CHECKSUMS.json`), reusable record/store validator (accounting, feasibility, incumbent correspondence, monotone histories, duplicate ids, references, runtimes, metadata, missing/duplicate records), original-vs-derived distinction (item 14) |
| `sensitivity.py` | Explicit named scenarios (dotted-path overrides), changed-vs-fixed recording, no unrestricted Cartesian product, pilot-first (item 25) |
| `reporting.py` | Markdown reports regenerated from stored data only (scope, config, benchmarks, algorithms, budgets, references, results, uncertainty, runtime, failures/omissions, limitations, evidence locations) (item 27) |
| `figures.py` | Standardized figures: convergence vs successful/attempted evals + wall-clock, gap distributions, paired differences, threshold success, runtime decomposition, candidate efficiency, performance vs dimension, sensitivity; units/legends/sample sizes; generic-synthetic labels; approximate references marked (item 28) |
| `simulator_adapter.py` | Item 29: software-interface validation ONLY — mock providers honoring the M4 contract (structured records, non-mutating snapshots, failure semantics); no physics activated |

**M7 validation experiment (item 26; matrix reported before execution):**
benchmark_a (2-D) + constrained_mixed_8d × {random_search, joint_bo, alternating_bo,
pairwise_bo, reference} × seeds 0–2, budget 40 (initial 10). Result: 20 completed,
2 skipped with recorded reasons (benchmark_a has no recommended block groupings),
all validity checks green, shared initialization verified, report + figures
generated, and a re-invocation of the runner performed **0 objective evaluations**
(resume without repetition). `results/m7_validation_experiment/` (regenerable).
Purpose: infrastructure validation only — no algorithm conclusions drawn.

**M7 sensitivity pilot (item 25):** explicit 3-scenario design on benchmark_a
(pool 100 / initial design 4 / budget 15; everything else fixed, seeds 0–1).
Only infrastructure validated; differences are descriptive
(`research_results/m7_sensitivity_pilot.json`).

**Statistical methodology (items 19–22)** — full paired summaries in
`research_results/m7_paired_comparative_summary.json` (derived). Headline
descriptive results (10 seeds; NOT superiority claims):

| Benchmark | Pair | Mean paired diff | W/L/T | d_z | Bootstrap 95% CI | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|
| separable_16d | pairwise − joint | +59.00 | 10/0/0 | 6.23 | [53.8, 64.8] | 0.0020 (0.0039) |
| separable_8d | pairwise − joint | +12.85 | 10/0/0 | 3.95 | [10.9, 14.8] | 0.0020 (0.0039) |
| coupled_8d | pairwise − joint | +22.30 | 10/0/0 | 3.73 | [18.7, 25.8] | 0.0020 (0.0059) |
| coupled_8d | alternating − joint | +17.54 | 10/0/0 | 2.97 | [14.0, 20.9] | 0.0020 (0.0059) |
| separable_4d | alternating − joint | +1.16 | 10/0/0 | 1.82 | [0.81, 1.54] | 0.0020 (0.0059) |
| constrained_mixed_8d | pairwise − joint | +0.21 | 9/1/0 | 0.93 | [0.08, 0.36] | 0.0273 (0.0547) |
| block_stagnation_4d | pairwise − joint | +0.31 | 8/2/0 | 0.90 | [0.11, 0.51] | 0.0273 (0.0547) |
| M5 A | BO − random search | +0.76 | 7/1/2 | — | — | — |
| M5 B | BO − random search | +0.38 | 8/2/0 | — | — | — |
| M5 C | BO − random search | +0.27 | 8/1/1 | — | — | — |

Limitations: ten seeds, single problem family per comparison; bootstrap
intervals are descriptive; Wilcoxon assumptions (approximate symmetry of paired
differences) are not verifiable at n = 10; family-wise control via Holm across
the presented per-benchmark family; practical significance is judged from effect
sizes and domain reasoning, never from p-values alone.

**Runtime accounting conventions (item 23):** five disjoint stage timers + explicit
unattributed overhead; never claimed to sum exactly to wall-clock. M5/M6 archives
use the NESTED convention (generation included feasibility) — documented; M7+
records use disjoint boundaries.

**Milestone reconciliation (item 33):** the original implementation guide has
EIGHT milestones; the instruction stream carries NINE (M9 = final documentation).
Instruction-stream M7 (experimental framework) maps onto roadmap M8 ("runner
package") plus parts of roadmap M7 ("analysis extensions"). No historical
artifacts were renamed; existing archives keep their M5/M6 labels. This is the
explicitly identified additional milestone.

### M8A scientific model review (instruction-stream item; STOP before M8B)

Scope: review only — NO application-specific optimization, NO physical radar
power-allocation implementation, NO corridor sensing optimization, NO large-scale
application experiments. The only code change is the backward-compatible
shared-initial-design interface (below) plus two new provenance entries and the
documentation set.

**Evidence reconciliation:**
- Final M7 regression: 668 passed (676 s) re-confirmed on committed `371952a`.
- Run counts: the M6 full-matrix archive holds exactly 220 records (160 primary =
  4 benchmarks × 4 methods × 10 seeds + 60 dimension-study = 2 × 3 × 10;
  alternating deliberately omitted from the dimension study) and the pilot archive
  32 records. The narrative figure "230 runs" was an arithmetic/reporting error —
  corrected via `experiment.m8a_run_count_reconciliation`; archives untouched.
- Profiling findings preserved: candidate generation 75–80% of wall, GP fit 20–23%,
  acquisition <1%; "GP fitting dominates" stays RETRACTED; no efficiency inference
  from acquisition-search dimensionality alone.

**Shared initial designs (validated immutable interface; fairness by construction):**
- Engine: `OptimizationConfig.initial_design_configs` — fail-fast ValueError on
  count/dimension/finiteness/feasibility; immutable tuple snapshot; explicit designs
  evaluated exactly in order; budget accounting unchanged; pool report notes explicit
  mode with zero sampling cost. Default `None` preserves M5/M6 behavior bit-for-bit
  (frozen trajectory 9.905619188792086 reproduced).
- Framework: `InitialDesignSettings.mode='explicit_shared'` — one design per
  (benchmark, dimension, seed) generated from a fresh problem and passed unchanged to
  every applicable method; infeasible design → RuntimeError before any run.
- 11 regression tests in `radar_bo/tests/test_shared_initial_design.py` (all pass).

**Provenance and documents:**
- `provenance_register.json`: 81 entries (17 unresolved preserved as `None`; NO
  missing value replaced by an arbitrary constant; illustrative values always labeled
  synthetic). Generated review companion: `docs/PARAMETER_RESOLUTION_REGISTER.md`
  (8-class taxonomy, per-unresolved-parameter resolution paths, extended-version
  cross-reference, parameters-requiring-approval list).
- `docs/` deliverables: `M8A_SCIENTIFIC_MODEL_REVIEW.md` (umbrella summary + stop
  statement), `MATHEMATICAL_MODEL_SPECIFICATION.md` (every implemented equation mapped
  to paper/proposed/synthetic + dimensional findings),
  `LITERATURE_AND_SOURCE_REGISTER.md` (verified citations; inaccessible sources marked
  unavailable; nothing fabricated), `ASSUMPTIONS_AND_LIMITATIONS.md` (paper facts vs
  proposed assumptions vs settings vs synthetic constructs; accepted resolutions
  preserved; tracking-state/scanning-metric/resource/β-scaling reviews; honest BO
  suitability assessment), `APPLICATION_VALIDATION_PLAN.md` (stages V1–V6 with
  model-rejection conditions; software vs scientific validation separated),
  `RESEARCH_GAP_ASSESSMENT.md` (established lines, our combination, non-claims).

**Extended-version finding (2026-11-26):** the radar-paper authors' extended journal
version (arXiv 2606.05526) was fetched and cross-checked. Its Table III reports
DIFFERENT values from the ICC Table II values reproduced here (σ²_{r,0} 16 vs 100,
σ²_{θ,0} 1e-6 vs 4e-4, σ²_w 16 vs 25, β 3e4 vs 1e6, Θ_max 0.9 vs 1) and replaces the
ICC scanning metric (Γ area ratio) with a missed-target count; SNR0 is absent in BOTH
versions (unresolved status re-confirmed). Its P_f = 1e-3 and P₀ = diag([1e4, 1e4,
1e2, 1e2]) are recorded as candidate values requiring separate review and approval —
not adopted. This repository reproduces the ICC values and does not silently switch.

**Proposed decisions documented but NOT applied (each requires separate review +
approval):** β normalization plan; first-two-observations velocity estimator;
extended-version candidate values; fixed-altitude/3-D state requirements;
corridor-metric requirements; a deterministic direct-search control for future
comparisons.

### Current limitations

- M3+M4 validate numerical and pipeline correctness only: all experiment artifacts are synthetic-validation records, NOT reproductions of the papers' experiments and NOT operational surveillance capability. The single-run M3 RMS position error is a numerical-validation result, NOT evidence of statistical filter consistency (no consistency analysis, e.g. NEES/NIS, has been performed).
- SNR0, all scanning physics parameters, the paper's S0, and task utility weights (uniform default) remain `None`/proposed where applicable; the synthetic validations supply explicitly labeled synthetic values to the components that need them, while the corresponding config fields stay unresolved.
- The synthetic scanning model is NOT validated radar physics; its outputs must never be presented as coverage, detectable range or detection probability results.
- The M4 tracking-cost provider is diagnostic and allocation-independent (paper Eq. (14) on current posteriors); a candidate's effect on FUTURE tracking requires simulation advancement (M5+).
- Dwell time is fixed at the paper reference tau0 = 1 s in the M3 validation; dwell allocation enters M4 only through the resource interfaces (no optimization).
- The `paper_zero_identity` initialization cannot run a full tracking episode from the origin (by design, diagnostic mode).
- M5 limitations: the generic engine implements the SINGLE-candidate JOINT mode only (no batching, no alternating/coordinate modes); acquisition supports conventional LogEI for low-noise/deterministic objectives (no noisy qLog variant is exercised); the GP likelihood noise is INFERRED (never forced to zero) even for deterministic benchmarks; the M5 validation budgets (100 evaluations, initial design 30) are provisional synthetic settings, not paper-derived; the seed 0-9 comparison is descriptive over 10 seeds and makes no superiority claim; BO results that fail to beat random search on some seeds are reported as measured; termination at a budget is not proof of global convergence; benchmark C's reference is numerical_approximate and may be exceeded legitimately.
- M6 limitations: the block architectures use ONE deterministic cyclic scheduler (priority/adaptive scheduling is future work); the improvement-only current-configuration acceptance policy can stall block-restricted search (demonstrated on the stagnation LP — this is honest evidence, not a defect); pair-wise and alternating coincide structurally on problems whose recommended groupings are identical (e.g. separable_4d) and their results must then be read as one architecture (identical numbers there); convergence curves use the per-evaluation incumbent history recorded by the full matrix (exact placement); block-restricted search on tightly constrained problems pays a large candidate-GENERATION cost (mostly-infeasible sampling, rejections never repaired) that can exceed the GP cost it was meant to save; the M6 comparison is descriptive over 10 seeds on synthetic problems only; the constrained-mixed reference is numerical_approximate; the discrete 2-variable block exhausts after 16 distinct combinations and then contributes only skips; figures are regenerable artifacts (not tracked) while the underlying per-seed data IS tracked under research_results/.
- M6 archived-data caveat (M7 review finding): random-search rows in the M6 archive store the per-evaluation objective SEQUENCE under the key `incumbent_history` (label wrong, data intact — best == max verified); the archived candidate-generation timer includes nested feasibility validation (never add the two fields); corrected semantics live in M7 runner records and derived artifacts.
- M7 limitations: the framework verifies shared initialization post hoc from stored records instead of injecting externally generated designs into the engine (the engine's seeded sampler IS the shared design; injection would require engine API changes and is documented as a deviation) — ADDRESSED in M8A: the engine now accepts validated immutable `initial_design_configs` and the framework `explicit_shared` mode injects them (the post-hoc verification remains available for pre-M7 archives); attempted-evaluation and wall-clock convergence axes are exact ONLY for M7 runner records (pre-M7 archives lack per-iteration timestamps — the axes are reported as unavailable there, never interpolated); Wilcoxon/bootstrap uncertainty at n = 10 is descriptive and assumption-limited; the M7 validation matrix (20 runs) validates INFRASTRUCTURE, not algorithms; sensitivity pilots cover three single-factor scenarios on one benchmark only; the simulator-interface validation uses mock providers (no physical sensing model activated, by design); `results/` artifacts from M6 were lost once to a sandbox reset and are regenerable (the durable tracked archives + checksums are the source of truth).
- M8A limitations: the review is documentation + the one backward-compatible interface change only — NO proposed decision was applied (β normalization, initialization candidates, extended-version values, 3-D/corridor requirements all remain documented proposals requiring approval); the literature review relies on arXiv extended versions because IEEE full texts are inaccessible from this environment, and the ICC paper's own reference list is unavailable (the item-7 review uses the extended version's bibliography with the citing paper's own attribution mappings); the paper extracts were lost in the sandbox reset, so Table II facts rest on the earlier extraction as recorded in the provenance register, cross-checked where possible against the fetched extended version; the provenance-register classification of historical entries could not be re-derived from primary sources for every entry and is labeled accordingly.
- The PyPI torch wheel bundles CUDA runtime libraries (~4 GB installed) although only CPU execution is used; this is a wheel-availability artifact, not a GPU requirement.
- Ambiguity handling (switchable, documented defaults): process covariance multiplier (`table_variance` = 25 default — a reading of Table II units/prose, NOT an author-confirmed correction; `equation_literal` = 625), initialization policy (`paper_zero_identity` diagnostic default / `measurement_driven` operational / `experimental_custom`), covariance update (`paper_equation_13` default / `joseph`), time-budget semantics (`discounted_cumulative` paper default; `hard_per_slot` proposed BO profile — implemented in M4 and never treated as equivalent to the paper semantics).


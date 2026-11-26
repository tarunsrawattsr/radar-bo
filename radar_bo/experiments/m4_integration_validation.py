"""Synthetic multi-task M4 integration validation (M4 instruction 16).

RESULT LABEL: synthetic M4 integration validation — software-pipeline
correctness only. This experiment does NOT demonstrate physical UAV detection
performance, validated radar coverage, or any optimization gains (no optimizer
exists in M4).

Pipeline exercised:

1. Configuration (validated nested SimulationConfig, synthetic labels).
2. Authoritative SimulationState with three synthetic tasks (truth stays in
   the simulation layer).
3. Frozen estimator-facing snapshot (truth excluded by design).
4. Candidate allocations covering all feasibility classes:
   partial budget / exact budget boundary / over-budget (hard violation) /
   feasible-but-distinct mixtures.
5. Feasibility classification under BOTH budget semantics (proposed hard
   per-slot vs paper-reference discounted cumulative — reported separately,
   never treated as equivalent).
6. Tracking-cost retrieval through the established interface (paper Eq. (14)
   diagnostic provider on the snapshot posteriors — allocation-independent by
   design; a candidate's effect on FUTURE tracking needs simulation
   advancement, which is out of M4 scope).
7. Synthetic scanning performance (linear response, labeled synthetic) fed by
   Eq. (24) remaining-time accounting.
8. Eq. (23) aggregation: U = -sum_n w_n c_n + beta * Gamma (beta = 1e6 paper
   value; uniform weights are the documented proposed default).
9. Non-mutation guarantees: snapshot integrity re-verified per evaluation;
   authoritative state fingerprint unchanged; physical time not advanced.
10. Order invariance under the declared common-random-numbers policy:
    candidates evaluated in a different order produce identical results.

Run::

    python -m radar_bo.experiments.m4_integration_validation [output_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from radar_bo.config import SimulationConfig, build_default_provenance
from radar_bo.config.resource_config import ResourceConfig
from radar_bo.config.tracking_config import TrackingConfig
from radar_bo.environment.target_state import Observation, TargetState
from radar_bo.optimization.allocation import ResourceAllocation
from radar_bo.optimization.constraints import ConstraintManager
from radar_bo.optimization.objective import ObjectiveEvaluator, SnapshotDiagnosticTrackingCostProvider
from radar_bo.optimization.resource_accounting import ResourceAccountant
from radar_bo.radar.initialization import initialize_estimator
from radar_bo.radar.scanning_model import SyntheticLinearScanningModel
from radar_bo.simulation.simulation_state import SimulationState

RESULT_LABEL = "synthetic M4 integration validation - pipeline correctness only; no physical detection or optimization claims"

N_TASKS = 3
T0 = 2.5  # paper revisit interval (single authoritative source: TrackingConfig)
SYNTHETIC_INIT_OBSERVATION = (3000.0, 0.5)  # shared synthetic initialization observation (range m, azimuth rad)


def build_config() -> SimulationConfig:
    tracking = SimulationConfig().tracking.model_copy(
        update={
            "initialization_policy": "measurement_driven",
            "measurement_driven_initial_velocity": (0.0, 0.0),  # SYNTHETIC prior
            "measurement_driven_initial_velocity_variance": 25.0,  # SYNTHETIC prior variance
        }
    )
    environment = SimulationConfig().environment.model_copy(
        update={
            "num_targets": N_TASKS,
            "initial_conditions": [
                (3000.0 * np.cos(0.5), 3000.0 * np.sin(0.5), 0.0, 0.0)
                for _ in range(N_TASKS)
            ],
        }
    )
    return SimulationConfig(tracking=tracking, environment=environment)


def run(output_dir: str | Path = "results/m4_integration_validation") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config()

    # ---- 1-2. Authoritative state: truth held here, never in snapshots --------
    observation = Observation(vector=SYNTHETIC_INIT_OBSERVATION, covariance=np.diag([1.0, 4.0e-6]))
    posteriors = tuple(initialize_estimator("measurement_driven", config.tracking, observation) for _ in range(N_TASKS))
    targets = tuple(
        TargetState(vector=np.array([SYNTHETIC_INIT_OBSERVATION[0] * np.cos(SYNTHETIC_INIT_OBSERVATION[1]),
                                     SYNTHETIC_INIT_OBSERVATION[0] * np.sin(SYNTHETIC_INIT_OBSERVATION[1]), 0.0, 0.0]),
                    target_id=f"task_{k}")
        for k in range(N_TASKS)
    )
    state = SimulationState(
        scenario_id="synthetic_m4_integration",
        physical_time_s=0.0,
        revisit_index=0,
        task_ids=tuple(f"task_{k}" for k in range(N_TASKS)),
        targets=targets,
        posteriors=posteriors,
        rng=np.random.default_rng(2024),
    )
    fingerprint_before = state.fingerprint()
    snapshot = state.snapshot()

    # ---- 3-4. Candidates covering all feasibility classes ----------------------
    action_set = list(config.resource.dwell_time_action_set)
    candidates = {
        "partial_budget": ResourceAllocation.from_dwell_indices([0, 2, 4], action_set),   # 0.9 s
        "mixed_budget": ResourceAllocation.from_dwell_indices([2, 5, 7], action_set),     # 0.3+0.6+0.8 = 1.7 s
        "exact_budget": ResourceAllocation.from_dwell_indices([4, 9, 9], action_set),     # 2.5 s (boundary)
        "over_budget": ResourceAllocation.from_dwell_indices([9, 9, 9], action_set),      # 3.0 s (> T0)
    }

    # ---- 5. Feasibility under BOTH semantics (reported separately) -------------
    # The default ResourceConfig semantics is the paper-reference
    # 'discounted_cumulative'; the PROPOSED hard per-slot profile is built
    # explicitly for the BO configuration (per M2/M4 review).
    resource_profile = config.resource.model_copy(update={"time_budget_semantics": "hard_per_slot"})
    hard_manager = ConstraintManager(resource_profile, N_TASKS, T0)
    paper_manager = ConstraintManager(
        config.resource.model_copy(update={"time_budget_semantics": "discounted_cumulative"}), N_TASKS, T0
    )
    feasibility = {
        name: {
            "proposed_hard_per_slot": hard_manager.check_allocation(a).model_dump(),
            "paper_reference_discounted_cumulative": paper_manager.check_allocation(a).model_dump(),
        }
        for name, a in candidates.items()
    }

    # ---- 6-8. Evaluator: paper Eq. (14) diagnostic costs + synthetic scanning ---
    # The evaluator runs on the PROPOSED hard per-slot profile; the paper-
    # reference classification above is reported alongside, never merged.
    evaluator = ObjectiveEvaluator(
        constraint_manager=hard_manager,
        accountant=ResourceAccountant(resource_profile, revisit_interval_s=T0),
        tracking_provider=SnapshotDiagnosticTrackingCostProvider(),
        scanning_model=SyntheticLinearScanningModel(
            synthetic_area_rate_m2_per_s=2000.0, synthetic_reference_area_m2=4000.0
        ),
        resource_config=resource_profile,
    )
    results = {name: evaluator.evaluate_candidate(snapshot, a) for name, a in candidates.items()}

    # ---- 9. Non-mutation guarantees ---------------------------------------------
    integrity_flags = {name: bool(r.snapshot_integrity_verified) for name, r in results.items()}
    state_preserved = state.fingerprint() == fingerprint_before
    time_preserved = state.physical_time_s == 0.0 and state.revisit_index == 0

    # ---- 10. Order invariance (CRN policy) ---------------------------------------
    shuffled_ids = ["over_budget", "partial_budget", "exact_budget", "mixed_budget"]
    shuffled_results = {name: evaluator.evaluate_candidate(snapshot, candidates[name]) for name in shuffled_ids}
    order_invariant = all(
        results[name].utility == shuffled_results[name].utility
        and results[name].per_task_tracking_costs == shuffled_results[name].per_task_tracking_costs
        and results[name].scanning_gamma == shuffled_results[name].scanning_gamma
        for name in candidates
    )

    # ---- Hand-verifiable numerical examples --------------------------------------
    # mixed_budget: tau = (0.3, 0.6, 0.8), sum = 1.7 s. Expected values follow
    # the paper formulas INDEPENDENTLY: Eq. (24) remaining time is
    # T0 * max(0, 1 - sum(tau)/T0) (this exact expression, not T0 - sum),
    # gamma = k * tau_s / S0 (Eq. (21) via the synthetic model constants),
    # utility = -sum(costs) + beta * gamma (Eq. (23)).
    diag_costs = [float(snapshot.posterior_covariance(k)[0, 0] + snapshot.posterior_covariance(k)[1, 1]) for k in range(N_TASKS)]
    beta = resource_profile.tracking_scanning_tradeoff
    mixed = results["mixed_budget"]
    mixed_remaining_expected = T0 * max(0.0, 1.0 - (0.3 + 0.6 + 0.8) / T0)  # paper Eq. (24)
    mixed_gamma_expected = 2000.0 * mixed_remaining_expected / 4000.0  # synthetic response + Eq. (21)
    mixed_utility_expected = -sum(diag_costs) + beta * mixed_gamma_expected  # Eq. (23)
    hand_checks = {
        "mixed_budget": {
            "expected_remaining_time_s": mixed_remaining_expected,
            "expected_gamma": mixed_gamma_expected,
            "expected_utility": mixed_utility_expected,
            "reported_utility": mixed.utility,
            "reported_gamma": mixed.scanning_gamma,
            "matches": bool(mixed.scanning_gamma == mixed_gamma_expected and mixed.utility == mixed_utility_expected),
        },
        "exact_budget": {
            "expected_remaining_time_s": 0.0,
            "expected_gamma": 0.0,
            "expected_utility": -sum(diag_costs),
            "reported_utility": results["exact_budget"].utility,
            "matches": bool(
                results["exact_budget"].scanning_gamma == 0.0
                and results["exact_budget"].utility == -sum(diag_costs)
            ),
        },
    }

    metrics = {
        "result_label": RESULT_LABEL,
        "num_tasks": N_TASKS,
        "revisit_interval_s": T0,
        "synthetic_labels": {
            "tracking_provider": "snapshot_diagnostic_eq14 (paper Eq. (14) on snapshot posteriors; allocation-independent by design)",
            "scanning_model": "synthetic_linear_scanning_v1 (SYNTHETIC response, not validated radar physics)",
            "velocity_prior": "synthetic (0, 0) m/s with variance 25 (m/s)^2",
        },
        "feasibility_classification": {
            name: {
                "hard_per_slot_feasible": feasibility[name]["proposed_hard_per_slot"]["is_feasible"],
                "paper_semantics_feasible": feasibility[name]["paper_reference_discounted_cumulative"]["is_feasible"],
                "normalized_usage": feasibility[name]["proposed_hard_per_slot"]["normalized_time_usage"],
            }
            for name in candidates
        },
        "evaluations": {
            name: {
                "status": r.status,
                "utility": r.utility,
                "aggregate_weighted_tracking_cost": r.aggregate_weighted_tracking_cost,
                "scanning_gamma": r.scanning_gamma,
                "remaining_scanning_time_s_eq24": r.resource_usage.remaining_scanning_time_s_paper_eq24 if r.resource_usage else None,
                "raw_time_deficit_s": r.resource_usage.raw_time_deficit_s if r.resource_usage else None,
                "allocation_id": r.allocation_id,
                "snapshot_integrity_verified": r.snapshot_integrity_verified,
            }
            for name, r in results.items()
        },
        "non_mutation_guarantees": {
            "snapshot_integrity_all_verified": all(integrity_flags.values()),
            "authoritative_state_fingerprint_preserved": state_preserved,
            "physical_time_not_advanced": time_preserved,
        },
        "order_invariance_under_crn": order_invariant,
        "hand_checks": hand_checks,
        "not_physical_performance": True,
        "not_optimization_gains": True,
    }

    # ---- Persist artifacts --------------------------------------------------------
    config.save_json(output_dir / "config.json")
    build_default_provenance(config).save_json(output_dir / "provenance_register.json")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (output_dir / "allocations.json").write_text(
        json.dumps({name: json.loads(a.model_dump_json()) for name, a in candidates.items()}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_results.json").write_text(
        json.dumps({name: json.loads(r.model_dump_json()) for name, r in results.items()}, indent=2, default=str),
        encoding="utf-8",
    )
    return metrics


def pytest_approx(value: float, rel: float = 1e-9) -> float:
    """Tolerance helper mirroring pytest.approx semantics for the JSON report."""
    return value  # comparisons below use exact arithmetic on these hand-built numbers


def main(argv: list[str]) -> int:
    output_dir = Path(argv[1]) if len(argv) > 1 else Path("results/m4_integration_validation")
    metrics = run(output_dir)
    print(RESULT_LABEL)
    print(json.dumps(metrics, indent=2, default=str))
    print(f"\nArtifacts written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

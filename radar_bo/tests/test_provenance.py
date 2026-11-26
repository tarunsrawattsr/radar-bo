"""Tests for the provenance register: source separation, unresolved preservation,
clear failure on unresolved values, and consistency with the default config."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.config.provenance import ProvenanceEntry, ProvenanceRegister, build_default_provenance
from radar_bo.config.scanning_config import RESOLVABLE_FIELDS
from radar_bo.config.simulation_config import SimulationConfig


@pytest.fixture
def register():
    return build_default_provenance()


def flagged(entry) -> str:
    """Rationale + notes joined, for text-presence assertions."""
    return " ".join([entry.rationale or "", *entry.notes])


class TestPaperEntries:
    def test_table_ii_values_recorded(self, register):
        e = register.get("tracking.reference_range_variance")
        assert e.value == 100.0 and e.unit == "m^2" and e.source == "paper_lu_2024"
        e = register.get("tracking.reference_angle_variance")
        assert e.value == 4.0e-4 and e.unit == "rad^2"
        e = register.get("tracking.revisit_interval")
        assert e.value == 2.5 and e.unit == "s"
        e = register.get("resource.tracking_scanning_tradeoff")
        assert e.value == 1.0e6

    def test_snr0_recorded_as_unresolved(self, register):
        """SNR0 is required by Eq. (5) but absent from the paper."""
        e = register.get("tracking.reference_snr")
        assert e.value is None
        assert e.source == "paper_lu_2024"
        assert "not reported" in (e.rationale or "").lower()

    def test_all_scanning_parameters_unresolved_with_units(self, register):
        for name in RESOLVABLE_FIELDS:
            e = register.get(f"scanning.{name}")
            assert e.value is None, f"scanning.{name} must stay unresolved by default"
            assert e.unit, f"scanning.{name} must carry a unit string"
            assert e.source == "paper_lu_2024"

    def test_standard_constant_separated(self, register):
        e = register.get("scanning.boltzmann_constant")
        assert e.value == 1.380649e-23
        assert e.source == "standard_constant"

    def test_ambiguities_are_flagged_in_notes_or_rationale(self, register):
        man = register.get("tracking.maneuverability_noise")
        # M4 review terminology: multiplier-based, unambiguous.
        assert "TERMINOLOGY" in flagged(man)
        assert "multiplier 625" in flagged(man)
        assert "multiplier 25" in flagged(man)
        assert "DEGENERATE" in flagged(register.get("tracking.initialization_policy"))
        semantics = register.get("tracking.measurement_variance_semantics")
        assert "DECREASES" in flagged(semantics)
        assert any("EXTRACTION ARTIFACT" in n for n in semantics.notes)

    def test_m3_review_entries(self, register):
        """M3 review outcomes are recorded: operational init policy, covariance
        update option, unresolved velocity priors, hard_per_slot reservation."""
        init = register.get("tracking.initialization_policy")
        assert "DIAGNOSTIC" in flagged(init)
        assert "OPERATIONAL" in flagged(init)
        assert register.get("tracking.measurement_driven_initial_velocity").value is None
        assert register.get("tracking.measurement_driven_initial_velocity_variance").value is None
        cov = register.get("tracking.covariance_update_method")
        assert cov.source == "proposed_assumption"
        assert "paper_equation_13" in (cov.rationale or "")
        budget = register.get("resource.time_budget_semantics")
        assert any("M4+" in n for n in budget.notes)
        man = register.get("tracking.maneuverability_noise")
        assert any("NOT an author-confirmed correction" in n for n in man.notes)

    def test_benzaghta_entries_separated(self, register):
        assert register.get("optimization.acquisition_xi").source == "paper_benzaghta_2023"
        assert register.get("optimization.num_acquisition_candidates").source == "paper_benzaghta_2023"
        assert register.get("optimization.search_mode").source == "proposed_assumption"

    def test_evaluation_budget_unresolved_by_design(self, register):
        assert register.get("optimization.evaluation_budget").value is None


class TestRegisterBehaviour:
    def test_duplicate_names_rejected(self):
        reg = ProvenanceRegister()
        reg.register(ProvenanceEntry(name="x", value=1.0, source="experimental_setting"))
        with pytest.raises(ValidationError if False else ValueError, match="[Dd]uplicate"):
            reg.register(ProvenanceEntry(name="x", value=2.0, source="experimental_setting"))

    def test_unknown_name_raises_keyerror(self, register):
        with pytest.raises(KeyError):
            register.get("no.such.entry")

    def test_require_resolved_returns_values(self, register):
        vals = register.require_resolved("tracking.reference_range_variance", "tracking.revisit_interval")
        assert vals == {"tracking.reference_range_variance": 100.0, "tracking.revisit_interval": 2.5}

    def test_require_resolved_fails_clearly_on_unresolved(self, register):
        with pytest.raises(UnresolvedParameterError) as excinfo:
            register.require_resolved("tracking.reference_snr", "scanning.wavelength")
        msg = str(excinfo.value)
        assert "tracking.reference_snr" in msg
        assert "scanning.wavelength" in msg
        assert "linear SNR" in msg  # unit included
        assert "not reported" in msg  # rationale included

    def test_require_resolved_unknown_name_fails(self, register):
        with pytest.raises(KeyError):
            register.require_resolved("no.such.entry")

    def test_unresolved_names_listing(self, register):
        unresolved = register.unresolved_names()
        assert "tracking.reference_snr" in unresolved
        assert "scanning.peak_transmit_power" in unresolved
        assert "tracking.reference_range_variance" not in unresolved

    def test_register_reflects_config_values(self):
        """Values come from the passed config, not hard-coded copies."""
        cfg = SimulationConfig()  # defaults
        reg = build_default_provenance(cfg)
        assert reg.get("tracking.reference_distance").value == cfg.tracking.reference_distance
        assert reg.get("resource.discount_factor").value == cfg.resource.discount_factor


class TestSerialization:
    def test_json_round_trip(self, register, tmp_path):
        path = register.save_json(tmp_path / "prov.json")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert len(parsed["entries"]) == len(register.entries)
        names = [e["name"] for e in parsed["entries"]]
        assert "tracking.reference_snr" in names
        restored = ProvenanceRegister.model_validate_json(path.read_text(encoding="utf-8"))
        assert restored == register


class TestLayerSeparation:
    def test_config_package_never_imports_optimizer_or_torch(self):
        """README section 3: config/simulation must not import the optimizer;
        the config layer stays free of torch/botorch/gpytorch imports."""
        import ast
        from pathlib import Path

        config_dir = Path(__file__).resolve().parents[1] / "config"
        banned_roots = {"torch", "botorch", "gpytorch", "linear_operator"}
        banned_modules = {"radar_bo.optimization", "radar_bo.radar", "radar_bo.environment", "radar_bo.simulation"}
        for path in config_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {a.name.split(".")[0] for a in node.names}
                    assert not (roots & banned_roots), f"{path.name} imports banned root: {roots & banned_roots}"
                    names = {a.name for a in node.names}
                    assert not (names & banned_modules), f"{path.name} imports banned module: {names & banned_modules}"
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    assert root not in banned_roots, f"{path.name} imports banned root: {root}"
                    assert (node.module or "") not in banned_modules, (
                        f"{path.name} imports banned module: {node.module}"
                    )

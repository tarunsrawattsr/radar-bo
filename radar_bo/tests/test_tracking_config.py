"""Tests for TrackingConfig: paper defaults, ambiguity switches, validation."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from radar_bo.config.tracking_config import TrackingConfig


class TestPaperDefaults:
    def test_table_ii_values(self):
        """Defaults reproduce Lu et al. Table II (p. 2789) exactly."""
        t = TrackingConfig()
        assert t.reference_range_variance == 100.0
        assert t.reference_angle_variance == 4.0e-4
        assert t.maneuverability_noise == 25.0
        assert t.reference_distance == 3000.0
        assert t.reference_dwell_time == 1.0
        assert t.revisit_interval == 2.5

    def test_unresolved_values_are_none(self):
        """SNR0 is not reported in the paper; transmit power is an extension."""
        t = TrackingConfig()
        assert t.reference_snr is None
        assert t.reference_transmit_power is None

    def test_structural_dimensions(self):
        t = TrackingConfig()
        assert t.state_dimension == 4
        assert t.measurement_dimension == 2

    def test_paper_initialization(self):
        t = TrackingConfig()
        assert t.initial_state == (0.0, 0.0, 0.0, 0.0)
        assert np.allclose(t.initial_covariance, np.eye(4))
        assert t.initialization_policy == "paper_zero_identity"

    def test_dimensions_are_frozen(self):
        t = TrackingConfig()
        with pytest.raises(ValidationError):
            t.state_dimension = 6


class TestAmbiguitySwitches:
    def test_maneuverability_interpretation_default_and_alternative(self):
        """Both Eq. (3) interpretations are switchable; default documented."""
        assert TrackingConfig().maneuverability_noise_interpretation == "table_variance"
        alt = TrackingConfig(maneuverability_noise_interpretation="equation_literal")
        assert alt.maneuverability_noise_interpretation == "equation_literal"
        assert alt.maneuverability_noise == 25.0  # value unchanged; semantics differ

    def test_invalid_maneuverability_interpretation_rejected(self):
        with pytest.raises(ValidationError):
            TrackingConfig(maneuverability_noise_interpretation="variance")  # not a valid literal

    def test_measurement_variance_scaling_consolidated(self):
        """M3 review instruction 3.2: Eq. (6) has ONE semantics (division by SNR).

        The former paper_as_printed/proposed_inverse_snr switch was consolidated
        away; the field no longer exists.
        """
        t = TrackingConfig()
        assert not hasattr(t, "measurement_variance_scaling")

    def test_initialization_policy_switch(self):
        assert TrackingConfig().initialization_policy == "paper_zero_identity"
        alt = TrackingConfig(initialization_policy="measurement_driven")
        assert alt.initialization_policy == "measurement_driven"

    def test_covariance_update_method_default_and_alternative(self):
        """Default reproduces Eq. (13); 'joseph' is the documented stability option."""
        assert TrackingConfig().covariance_update_method == "paper_equation_13"
        alt = TrackingConfig(covariance_update_method="joseph")
        assert alt.covariance_update_method == "joseph"
        with pytest.raises(ValidationError):
            TrackingConfig(covariance_update_method="cholesky_update")

    def test_measurement_driven_init_assumptions_unresolved_by_default(self):
        """Velocity prior assumptions are proposed, unresolved (None) by default."""
        t = TrackingConfig()
        assert t.measurement_driven_initial_velocity is None
        assert t.measurement_driven_initial_velocity_variance is None

    def test_measurement_driven_init_assumptions_validation(self):
        t = TrackingConfig(
            measurement_driven_initial_velocity=(5.0, -2.0),
            measurement_driven_initial_velocity_variance=25.0,
        )
        assert t.measurement_driven_initial_velocity == (5.0, -2.0)
        assert t.measurement_driven_initial_velocity_variance == 25.0
        with pytest.raises(ValidationError):
            TrackingConfig(measurement_driven_initial_velocity_variance=-1.0)
        with pytest.raises(ValidationError):
            TrackingConfig(measurement_driven_initial_velocity_variance=0.0)


class TestValidation:
    def test_negative_variance_rejected(self):
        with pytest.raises(ValidationError):
            TrackingConfig(reference_range_variance=-1.0)

    def test_zero_angle_variance_rejected(self):
        with pytest.raises(ValidationError):
            TrackingConfig(reference_angle_variance=0.0)

    def test_dwell_exceeding_revisit_rejected(self):
        with pytest.raises(ValidationError, match="revisit_interval"):
            TrackingConfig(reference_dwell_time=3.0)  # > T0 = 2.5 s

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            TrackingConfig(nonexistent_field=1.0)

    def test_paper_policy_rejects_custom_initial_state(self):
        with pytest.raises(ValidationError, match="experimental_custom"):
            TrackingConfig(initial_state=(100.0, 0.0, 0.0, 0.0))

    def test_paper_policy_rejects_custom_covariance(self):
        with pytest.raises(ValidationError, match="experimental_custom"):
            TrackingConfig(initial_covariance=(2.0 * np.eye(4)).tolist())

    def test_experimental_custom_allows_overrides(self):
        t = TrackingConfig(
            initialization_policy="experimental_custom",
            initial_state=(1000.0, 2000.0, 10.0, -5.0),
            initial_covariance=(100.0 * np.eye(4)).tolist(),
        )
        assert t.initial_state[0] == 1000.0

    def test_nonsymmetric_covariance_rejected(self):
        bad = np.eye(4).tolist()
        bad[0][1] = 0.5  # asymmetric
        with pytest.raises(ValidationError, match="symmetric"):
            TrackingConfig(initial_covariance=bad)

    def test_non_psd_covariance_rejected(self):
        bad = np.diag([1.0, -1.0, 1.0, 1.0]).tolist()
        with pytest.raises(ValidationError, match="positive semidefinite"):
            TrackingConfig(initial_covariance=bad)

    def test_wrong_shape_covariance_rejected(self):
        with pytest.raises(ValidationError, match="4x4"):
            TrackingConfig(initial_covariance=np.eye(3).tolist())

    def test_assignment_revalidation(self):
        t = TrackingConfig()
        with pytest.raises(ValidationError):
            t.reference_distance = -5.0

"""Shared configuration-layer primitives.

This module (and the whole ``radar_bo.config`` package) must never import the
optimizer, radar or simulation layers (README section 3 layer separation).
Pydantic is used for validation and serialization only, never for numerical
matrix computation.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base class for all configuration models.

    * ``extra="forbid"`` — undeclared fields are rejected, so typos in a
      config file can never silently produce a wrong experiment.
    * ``validate_assignment=True`` — mutation of a populated config is
      re-validated, keeping invariants (positivity, bounds, shapes) intact.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class UnresolvedParameterError(RuntimeError):
    """Raised when an experiment requires a parameter that is still ``None``.

    The framework must never invent missing paper parameters (README
    non-negotiables); instead it fails clearly, listing every unresolved name
    and its unit, so the user can supply a justified value.
    """

    def __init__(self, missing: Mapping[str, str], guidance: str = "") -> None:
        self.missing: dict[str, str] = dict(missing)
        self.guidance = guidance
        lines = [f"  - {name} (unit: {unit or 'unspecified'})" for name, unit in sorted(missing.items())]
        message = (
            "Experiment requires unresolved parameter(s). The framework does not "
            "invent missing values (README non-negotiables). Missing:\n" + "\n".join(lines)
        )
        if guidance:
            message += f"\n{guidance}"
        super().__init__(message)


def ensure_resolved(values: Mapping[str, Any], guidance: str = "") -> None:
    """Raise :class:`UnresolvedParameterError` if any mapped value is ``None``."""
    missing = {name: "None" for name, value in values.items() if value is None}
    if missing:
        raise UnresolvedParameterError(missing, guidance=guidance)

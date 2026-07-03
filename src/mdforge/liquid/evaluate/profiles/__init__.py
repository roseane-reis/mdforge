"""Species profiles for liquid evaluation.

A *profile* encapsulates the species-specific knowledge the engine-agnostic
pipeline needs — which local atom is the ordering site (oxygen for water), how
many atoms per molecule, the element order, and the physical partial charges.
Water is the first (and, today, only) profile; adding another liquid means
adding another profile with the same surface.
"""

from __future__ import annotations

from .water import WaterProfile, water_profile

__all__ = ["WaterProfile", "water_profile"]

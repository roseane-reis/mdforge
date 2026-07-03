"""Packaged reference data for liquid-model evaluation.

Loaded via :mod:`importlib.resources`. Files:

- ``water_298K.json`` — experimental + baseline-model property values at
  298.15 K / 1 atm, with source citations and scoring metadata.
- ``298_1_g{OO,OH,HH}.txt`` — Soper (2000) experimental partial RDFs at 298 K
  (``Bin no.  r  g(r)  std``; 4 header lines).
- ``SOURCES.md`` — human-readable provenance (not loaded at runtime).
"""

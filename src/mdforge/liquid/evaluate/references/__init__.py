"""Packaged reference data for liquid-model evaluation.

Loaded via :mod:`importlib.resources`. Files:

- ``water_298K.json`` — experimental + baseline-model property values at
  298.15 K / 1 atm, with source citations and scoring metadata.
- ``298_1_g{OO,OH,HH}.txt`` — Soper (2013, revised) experimental partial RDFs at
  298 K (``Bin no.  r  g(r)  std``; 4 header lines).
- ``skinner2014_gOO.txt`` — Skinner (2014) X-ray O-O RDF (second reference).
- ``SOURCES.md`` — human-readable provenance (not loaded at runtime).
"""

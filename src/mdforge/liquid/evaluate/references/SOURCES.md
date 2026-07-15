# Water reference data — provenance

All values in `water_298K.json` are for **liquid water at 298.15 K / 1 atm**.
The "good vs bad" quality bar is the **TIP3P** model; "excellent" is within 1 %
of experiment (or within the stated experimental uncertainty).

## Experimental values and the TIP3P / SPC-E / TIP3P-FB / OPC3 baselines

Izadi, S.; Onufriev, A. V. *Accuracy limit of rigid 3-point water models.*
J. Chem. Phys. **145**, 074501 (2016). DOI: 10.1063/1.4960175 — **Table III**
(298.16 K, 1 bar) is the source for density, ΔHvap, ε₀, D, Cp, κ_T, α_T and the
TIP3P/SPC-E/TIP3P-FB/OPC3 model columns.

## HIPPO baseline

Rackers, J. A.; Silva, R. R.; Wang, Z.; Ponder, J. W. *A Polarizable Water
Potential Derived from a Model Electron Density.* J. Chem. Theory Comput.
**17**, 7056–7084 (2021). DOI: 10.1021/acs.jctc.1c00628 — Table 4 and the
structure/diffusion figures. HIPPO D (2.557 ×10⁻⁵ cm²/s) is Yeh–Hummer
finite-size corrected; g_OO first peak ≈ 2.785 Å, height ≈ 3.0.

## Structural experimental references (g_OO, q, H-bonds, coordination)

The packaged `298_1_g{OO,OH,HH}.txt` partial RDFs (columns `Bin no.  r  g(r)  std`,
4 header lines) are Soper's **revised** ambient-water RDFs:

Soper, A. K. *The Radial Distribution Functions of Water: Is There Anything We
Can Say for Sure?* ISRN Physical Chemistry **2013**, 279463 (2013).
DOI: 10.1155/2013/279463. (g_OO first peak ≈ 2.79 Å, height ≈ 2.50.)

The scalar structural values in `water_298K.json` attributed to this reference —
g_OO first peak 2.8 Å / height 2.5, tetrahedral order q ≈ 0.576, H-bonds/molecule
≈ 3.6, O–O coordination ≈ 4.5 — carry `source: "soper2013"`. Except g_OO peak
position (rated), these have **no TIP3P baseline** and are reported as `unrated`
(they can still earn "excellent" within 1 %).

## Second O–O reference (X-ray)

Skinner, L. B.; Benmore, C. J.; et al. *Benchmark oxygen-oxygen pair-distribution
function of ambient water from x-ray diffraction measurements with a wide Q-range.*
J. Chem. Phys. **138**, 074506 (2013); APS high-energy dataset (2014) — the
packaged `skinner2014_gOO.txt` (same `Bin no.  r  g(r)  std` layout). The
near-ambient **295.1 K** column (best-quality flowing-stream data) is used;
g_OO first peak ≈ 2.80 Å, height ≈ 2.57. Plotted alongside Soper (2013) as an
independent experimental O–O curve; not used for scoring.

## Notes on corrections

- **Self-diffusion** must be compared using the Yeh–Hummer finite-size-corrected
  value, not the raw PBC estimate.
- **ΔHvap** is classical (no nuclear-quantum correction). For a rigid
  fixed-charge model it is the apparent cohesive energy `-PE/molecule + RT`
  (gas-phase PE = 0); a strongly over-bound value flags an over-attractive model.

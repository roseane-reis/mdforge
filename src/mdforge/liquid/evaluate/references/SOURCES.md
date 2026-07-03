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

Soper, A. K. *The radial distribution functions of water and ice.* Chem. Phys.
**258**, 121–137 (2000). DOI: 10.1016/S0301-0104(00)00179-8 — the packaged
`298_1_g{OO,OH,HH}.txt` partial RDFs (columns `Bin no.  r  g(r)  std`, 4 header
lines) and the ambient-water structural values: tetrahedral order q ≈ 0.576,
H-bonds/molecule ≈ 3.6, O–O coordination ≈ 4.5, g_OO peak height ≈ 2.75. These
structural metrics have **no TIP3P baseline** in the cited papers and are
therefore reported as `unrated` (they can still earn "excellent" within 1 %).

## Notes on corrections

- **Self-diffusion** must be compared using the Yeh–Hummer finite-size-corrected
  value, not the raw PBC estimate.
- **ΔHvap** is classical (no nuclear-quantum correction). For a rigid
  fixed-charge model it is the apparent cohesive energy `-PE/molecule + RT`
  (gas-phase PE = 0); a strongly over-bound value flags an over-attractive model.

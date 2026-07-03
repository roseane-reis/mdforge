"""Tinker parameter (.prm) and key (.key) file read & write (goal c).

Ported from ``analyzetool.prmedit`` — the richest Tinker format library in the
legacy code, covering the HIPPO/AMOEBA term set (atom, bond, angle, strbnd,
opbend, torsion, imptors, ureybrad, vdw, charge, multipole, polarize, chgpen,
dispersion, repulsion, chgtrn, bndcflux/angcflux, exchpol).

``process_prm`` → structured ``prmdict``; ``write_prm`` is its inverse (a
write→reparse round-trip is stable). ``write_key`` emits a Tinker ``.key`` with
embedded simulation settings.

Changes from the source: per-class blocks (chgpen/dispersion/…) are parsed by
iterating the actual lines keyed by class id (the legacy positional
``range(nclas)`` indexing IndexErrors on partial blocks); the ``multipole_factors``
``.sum`` typo (a method comparison that never fired) is fixed.

System-building utilities (``combine_params``, ``update_types``, ``prm_from_key``)
are deferred to later phases.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np

# --- Section header comment blocks (written by write_prm) --------------------

_FORCEFIELD = "HIPPO-FORGE_DERIVED"

PRM_HEADER = f"""##############################
##                          ##
##  Force Field Definition  ##
##                          ##
##############################

forcefield              {_FORCEFIELD}

bond-cubic              -2.55
bond-quartic            3.793125
angle-cubic             -0.014
angle-quartic           0.000056
angle-pentic            -0.0000007
angle-sextic            0.000000022
opbendtype              ALLINGER
opbend-cubic            -0.014
opbend-quartic          0.000056
opbend-pentic           -0.0000007
opbend-sextic           0.000000022
torsionunit             0.5
dielectric              1.0
polarization            MUTUAL
d-equals-p
rep-12-scale            0.0
rep-13-scale            0.0
rep-14-scale            0.4
rep-15-scale            0.8
disp-12-scale           0.0
disp-13-scale           0.0
disp-14-scale           0.4
disp-15-scale           0.8
mpole-12-scale          0.0
mpole-13-scale          0.0
mpole-14-scale          0.4
mpole-15-scale          0.8
polar-12-scale          0.0
polar-13-scale          0.5
polar-14-scale          1.0
polar-15-scale          1.0
polar-12-intra          0.0
polar-13-intra          0.0
polar-14-intra          0.0
polar-15-intra          0.5
direct-11-scale         0.0
direct-12-scale         1.0
direct-13-scale         1.0
direct-14-scale         1.0
mutual-11-scale         1.0
mutual-12-scale         1.0
mutual-13-scale         1.0
mutual-14-scale         1.0
induce-12-scale         0.2
induce-13-scale         1.0
induce-14-scale         1.0
induce-15-scale         1.0


"""


def _hdr(title: str) -> str:
    bar = "#" * (len(title) + 8)
    return f"\n\n{bar}\n##  {title}  ##\n{bar}\n\n"


_SECTION_TITLES = {
    "atom": "Atom Type Definitions",
    "multipole": "Atomic Multipole Parameters",
    "polarize": "Dipole Polarizability Parameters",
    "vdw": "Van der Waals Parameters",
    "charge": "Atomic Partial Charge Parameters",
    "chgpen": "Charge Penetration Parameters",
    "dispersion": "Dispersion Parameters",
    "repulsion": "Repulsion Parameters",
    "chgtrn": "Charge Transfer Parameters",
    "bond": "Bond Stretching Parameters",
    "angle": "Angle Bending Parameters",
    "strbnd": "Stretch-Bend Parameters",
    "ureybrad": "Urey-Bradley Parameters",
    "opbend": "Out-of-Plane Bend Parameters",
    "torsion": "Torsional Parameters",
    "imptors": "Improper Torsional Parameters",
    "cflux": "Charge Flux Parameters",
    "exchpol": "Exchange Polarization",
}

# --- Tinker .key templates (write_key) ---------------------------------------

KEY_LIQUID = """parameters          {prm}
integrator respa

dcd-archive
tau-pressure      5.00
tau-temperature   1.0
barostat          langevin
volume-trial      5

digits            10
printout          500

a-axis            {a_axis}
cutoff            7
neighbor-list
ewald
dewald

polarization      mutual
polar-eps         1e-05
polar-predict     aspc
#########################
"""

KEY_GAS = """parameters          {prm}
integrator        stochastic

dcd-archive
tau-temperature   0.1
volume-scale      molecular
THERMOSTAT        BUSSI
BAROSTAT          MONTECARLO

digits            10
printout          5000

polarization      mutual
polar-eps         1e-06

fix-chgpen
#########################
"""


def compute_total_charge(mcharges: np.ndarray, factors: np.ndarray) -> float:
    return float((np.asarray(mcharges) * np.asarray(factors)).sum())


def multipole_factors(prmdict: dict[str, Any]) -> np.ndarray:
    """Find integer multipole charge-balance factors that neutralize the cell.

    Returns all-ones when the monopoles already sum to ~0.
    """
    charges = np.array([row[0] for row in prmdict["multipole"][1]], dtype=float)
    nmpol = len(charges)
    factors = np.ones(nmpol, dtype=int)
    if np.abs(charges.sum()) < 1e-5:
        return factors

    tested: list[list[int]] = []
    for combo in itertools.combinations_with_replacement(range(1, 20), nmpol):
        if np.array(combo).sum() == nmpol:  # fixed: legacy compared the method object
            continue
        for perm in itertools.permutations(combo):
            cand = list(perm)
            if cand in tested:
                continue
            if np.abs(compute_total_charge(charges, perm)) < 1e-5:
                return np.array(perm, dtype=int)
            tested.append(cand)
    return factors


def process_prm(prmfn: str | Path) -> dict[str, Any]:
    """Parse a Tinker ``.prm`` into a structured ``prmdict`` (HIPPO term set)."""
    prmfile0 = Path(prmfn).read_text().splitlines(keepends=True)
    prmfile = np.array([ln for ln in prmfile0 if ln and ln[0] != "#" and ln != "\n"])

    nstart = 0
    for k, line in enumerate(prmfile):
        if line[:6] == "atom  ":
            nstart = k
            break

    # multipole_factors directive (a comment line)
    mfactors: list[int] = []
    for line in prmfile0:
        if "multipole_factors" in line:
            seg = line.strip("\n").split("=")
            if len(seg) < 2:
                seg = line.strip("\n").split(":")
            for tok in seg[1].split(","):
                digits = "".join(c for c in tok if c.isdigit())
                if digits:
                    mfactors.append(int(digits))
            break

    bg = np.array([a[:5] for a in prmfile])

    def section(tag: str) -> np.ndarray:
        return prmfile[nstart:][bg[nstart:] == tag]

    atms = prmfile[bg == "atom "]
    bnd, angl, strb = section("bond "), section("angle"), section("strbn")
    opbe, tors, pol = section("opben"), section("torsi"), section("polar")
    cpen, disp, rep = section("chgpe"), section("dispe"), section("repul")
    ctrn, bflx, aflx = section("chgtr"), section("bndcf"), section("angcf")
    exchp = section("exchp")
    vdwt, chg = section("vdw  "), section("charg")
    itor, urey = section("impto"), section("ureyb")

    minds = np.where(bg == "multi")[0]
    mlines: list[str] = []
    if len(minds):
        mlines = [a for a in prmfile[minds[0]:minds[-1] + 5] if len(a) > 5 and a.split()[0] != "#"]

    # --- atoms ---
    atoms: list[list] = []
    atmtyp: list[int] = []
    tclasses: list[int] = []
    typcls: dict[int, int] = {}
    for lin in atms:
        t, cl = int(lin.split()[1]), int(lin.split()[2])
        typcls[t] = cl
        atmtyp.append(t)
        tclasses.append(cl)
        nm = lin.split('"')
        atoms.append(nm[0].split()[2:] + [nm[1]] + nm[2].split())

    tclasses = sorted(set(tclasses))
    clspos = {t: i for i, t in enumerate(tclasses)}
    nclas = len(tclasses)
    atmtyp = sorted(atmtyp)
    atmpos = {t: i for i, t in enumerate(atmtyp)}
    natms = len(atmtyp)

    # --- per-class blocks (iterate actual lines, keyed by class) ---
    chgpen = np.zeros((nclas, 2))
    dispersion = np.zeros(nclas)
    repulsion = np.zeros((nclas, 3))
    chgtrn = np.zeros((nclas, 2))
    charge = np.zeros(nclas)
    vdw = np.zeros((nclas, 2))

    for line in cpen:
        s = line.split()
        chgpen[clspos[int(s[1])]] = [float(s[2]), float(s[3])]
    for line in disp:
        s = line.split()
        dispersion[clspos[int(s[1])]] = float(s[2])
    for line in rep:
        s = line.split()
        repulsion[clspos[int(s[1])]] = [float(s[2]), float(s[3]), float(s[4])]
    for line in ctrn:
        s = line.split()
        chgtrn[clspos[int(s[1])]] = [float(s[2]), float(s[3])]
    for line in vdwt:
        s = line.split()
        vdw[clspos[int(s[1])]] = [float(s[2]), float(s[3])]
    for line in chg:
        s = line.split()
        charge[clspos[int(s[1])]] = float(s[2])

    # --- polarize (per type) ---
    polarize: list[list] = [[0] * natms, [0] * natms]
    for line in pol:
        s = line.split()
        i = atmpos[int(s[1])]
        polarize[0][i] = float(s[2])
        if len(s) > 3:
            itest = float(s[3])
            con = [int(a) for a in (s[3:] if itest.is_integer() else s[4:])]
            polarize[1][i] = con

    # --- multipoles (5-line blocks) ---
    multipoles: list[list] = [[], []]
    for i, line in enumerate(mlines[::5]):
        if "#" in line:
            line = line.split("#")[0]
        typs = [int(a) for a in line.split()[1:-1]]
        vals = [float(line.split()[-1])]
        for z, k in enumerate(mlines[i * 5 + 1:i * 5 + 5]):
            s = k.split()
            vals += [float(a) for a in (s[:-1] if z == 3 else s)]
        multipoles[0].append(typs)
        multipoles[1].append(vals)

    # --- bonded terms ---
    bond: list[list] = [[], [], []]
    for line in bnd:
        s = line.split()
        bond[0].append(s[1:3])
        bond[1].append(float(s[3]))
        bond[2].append(float(s[4]))

    angle: list[list] = [[], [], [], []]
    for line in angl:
        s = line.split()
        angle[0].append([int(b) for b in s[1:4]])
        angle[1].append(float(s[4]))
        angle[2].append(float(s[5]))
        angle[3].append(s[6] if len(s) > 6 else "")

    strbnd: list[list] = [[], [], []]
    for line in strb:
        s = line.split()
        strbnd[0].append(s[1:4])
        strbnd[1].append(float(s[4]))
        strbnd[2].append(float(s[5]))

    ureybrad: list[list] = [[], [], []]
    for line in urey:
        s = line.split()
        ureybrad[0].append(s[1:4])
        ureybrad[1].append(float(s[4]))
        ureybrad[2].append(float(s[5]))

    opbend: list[list] = [[], []]
    for line in opbe:
        s = line.split()
        opbend[0].append(s[1:5])
        opbend[1].append(float(s[5]))

    torsion: list[list] = [[], []]
    for line in tors:
        s = line.split()
        torsion[0].append(s[1:5])
        torsion[1].append([float(a) for a in s[5:]])

    imptors: list[list] = [[], []]
    for line in itor:
        s = line.split()
        imptors[0].append(s[1:5])
        imptors[1].append([float(a) for a in s[5:]])

    bndcflux: list[list] = [[], []]
    for line in bflx:
        s = line.split()
        bndcflux[0].append(s[1:3])
        bndcflux[1].append(float(s[3]))

    angcflux: list[list] = [[], []]
    for line in aflx:
        s = line.split()
        angcflux[0].append(s[1:4])
        angcflux[1].append([float(a) for a in s[4:]])

    exchpol_list = [[float(a) for a in line.split()[1:]] for line in exchp]
    exchpol = np.array(exchpol_list) if exchpol_list else np.array([])
    if exchpol.size:
        exchpol = exchpol[np.argsort(exchpol[:, 0])]

    return {
        "atom": atoms,
        "types": atmtyp,
        "typcls": typcls,
        "bond": bond,
        "angle": angle,
        "strbnd": strbnd,
        "ureybrad": ureybrad,
        "opbend": opbend,
        "torsion": torsion,
        "imptors": imptors,
        "charge": charge,
        "vdw": vdw,
        "chgpen": chgpen,
        "dispersion": dispersion,
        "repulsion": repulsion,
        "polarize": polarize,
        "bndcflux": bndcflux,
        "angcflux": angcflux,
        "chgtrn": chgtrn,
        "multipole": multipoles,
        "multipole_factors": mfactors,
        "exchpol": exchpol,
    }


def write_prm(prmdict: dict[str, Any], fnout: str | Path, mfactors: list[int] | None = None) -> str:
    """Write a ``prmdict`` (from :func:`process_prm`) back to a Tinker ``.prm``."""
    sorttypes = np.argsort(np.array(prmdict["types"]))
    out = PRM_HEADER + _hdr(_SECTION_TITLES["atom"]).lstrip("\n")

    # atoms
    typcls: dict[int, int] = {}
    for k in sorttypes:
        t = prmdict["types"][k]
        v = list(prmdict["atom"][k])
        acls = int(v[0])
        typcls[t] = acls
        name = v[2].strip('"')
        pad = 28 - len(name)
        out += (f"{'atom':11s} {t:3d}  {acls:3d}    {v[1]:6s}"
                f'"{name}"{" ":{pad}s} {int(v[3]):3d} {float(v[4]):9.3f} {int(v[5]):4d}\n')

    # multipole
    if len(prmdict["multipole"][0]) > 0:
        out += _hdr(_SECTION_TITLES["multipole"])
        factors = np.array(mfactors if mfactors else prmdict.get("multipole_factors", []), dtype=int)
        if factors.size == 0:
            factors = multipole_factors(prmdict)
        nmpol = factors.shape[0]
        lbl = "+".join((f"c{k + 1}" if factors[k] == 1 else f"{factors[k]}*c{k + 1}")
                       for k in range(nmpol - 1))
        for k, v in enumerate(prmdict["multipole"][1]):
            typs = prmdict["multipole"][0][k]
            if isinstance(typs, str):
                typs = typs.split()
            c = "".join(f"{int(ts):4d} " for ts in typs)
            first = f"{'multipole':11s}{c:<27s}{v[0]:10.6f}"
            if k == nmpol - 1 and nmpol >= 2:
                first += f" # -{lbl}" if nmpol == 2 else f" # -({lbl})"
                first += f"/{factors[k]}\n" if factors[k] > 1 else "\n"
            else:
                first += "\n"
            out += first
            out += f"{' ':38s}{v[1]:10.6f} {v[2]:10.6f} {v[3]:10.6f}\n"
            out += f"{' ':38s}{v[4]:10.6f}\n"
            out += f"{' ':38s}{v[5]:10.6f} {v[6]:10.6f}\n"
            out += f"{' ':38}{v[7]:10.6f} {v[8]:10.6f} {-(v[4] + v[6]):10.6f}\n"
        fac = f"[ {factors[0]}" + "".join(f",{fc}" for fc in factors[1:]) + " ]"
        out += f"## multipole_factors = {fac} \n"

    # polarize
    if np.sum(prmdict["polarize"][0]) != 0:
        out += _hdr(_SECTION_TITLES["polarize"])
        for k in sorttypes:
            t = prmdict["types"][k]
            v = prmdict["polarize"][0][k]
            con = prmdict["polarize"][1][k]
            c = "".join(f"  {int(ts):3d}" for ts in con) if isinstance(con, list) else ""
            out += f"{'polarize':16s} {t:<11d}{v:10.6f}{c}\n"

    # per-class scalar/vector blocks
    def class_block(term: str, fmt) -> str:
        if np.sum(prmdict[term]) == 0:
            return ""
        s = _hdr(_SECTION_TITLES.get(term, term))
        for k in sorttypes:
            t = typcls[prmdict["types"][k]]
            s += fmt(t, prmdict[term][k])
        return s

    out += class_block("vdw", lambda t, v: f"{'vdw':16s} {t:<11d}{v[0]:8.4f} {v[1]:12.6f}\n")
    out += class_block("charge", lambda t, v: f"{'charge':16s} {t:<11d}{v:10.6f}\n")
    out += class_block("chgpen", lambda t, v: f"{'chgpen':16s} {t:<11d}{v[0]:8.4f} {v[1]:12.6f}\n")
    out += class_block(
        "repulsion",
        lambda t, v: f"{'repulsion':16s} {t:<11d}{v[0]:10.6f} {v[1]:10.6f} {v[2]:10.6f}\n",
    )

    # dispersion carries the chgpen damping in column 2
    if np.sum(prmdict["dispersion"]) > 0:
        out += _hdr(_SECTION_TITLES["dispersion"])
        for k in sorttypes:
            t = typcls[prmdict["types"][k]]
            out += f"{'dispersion':16s} {t:<11d}{prmdict['dispersion'][k]:10.6f} {prmdict['chgpen'][k][1]:10.6f}\n"

    out += class_block(
        "chgtrn", lambda t, v: f"{'chgtrn':16s} {t:<11d}{v[0]:10.6f} {v[1]:10.6f}\n"
    )

    # bonded terms (keyed by class pairs/triples already stored)
    def typed_block(term: str, header_key: str, fmt) -> str:
        if len(prmdict[term][0]) == 0:
            return ""
        s = _hdr(_SECTION_TITLES.get(header_key, header_key))
        for k, v in enumerate(prmdict[term][1]):
            s += fmt(k, v)
        return s

    out += typed_block(
        "bond", "bond",
        lambda k, v: f"{'bond':12s}{_c(prmdict['bond'][0][k], 2):<16s}{v:10.6f} {prmdict['bond'][2][k]:10.6f}\n",
    )

    if len(prmdict["angle"][0]) > 0:
        out += _hdr(_SECTION_TITLES["angle"])
        for k, v in enumerate(prmdict["angle"][1]):
            c = _c(prmdict["angle"][0][k], 3)
            row = f"{'angle':12s}{c:<16s}{v:10.6f} {prmdict['angle'][2][k]:10.3f}"
            v3 = prmdict["angle"][3][k]
            out += (row + f" {v3:10s}\n") if v3 else (row + "\n")

    out += typed_block(
        "strbnd", "strbnd",
        lambda k, v: f"{'strbnd':12s}{_c(prmdict['strbnd'][0][k], 3):<16s}{v:10.6f} {prmdict['strbnd'][2][k]:10.6f}\n",
    )
    out += typed_block(
        "ureybrad", "ureybrad",
        lambda k, v: f"{'ureybrad':12s}{_c(prmdict['ureybrad'][0][k], 3):<16s}{v:10.6f} {prmdict['ureybrad'][2][k]:10.6f}\n",
    )
    out += typed_block(
        "opbend", "opbend",
        lambda k, v: f"{'opbend':12s}{_c(prmdict['opbend'][0][k], 4):<25s} {v:12.6f}\n",
    )

    if len(prmdict["torsion"][0]) > 0:
        out += _hdr(_SECTION_TITLES["torsion"])
        for k, v in enumerate(prmdict["torsion"][1]):
            ids = [int(a) for a in prmdict["torsion"][0][k]]
            if sum(ids) == 0:
                continue
            line = f"{'torsion':12s}{_c(ids, 4):<25s} "
            for i in range(0, len(v), 3):
                line += f"{v[i]:7.3f} {v[i + 1]:2.1f} {int(v[i + 2]):d}"
            out += line + "\n"

    if len(prmdict["imptors"][0]) > 0:
        out += _hdr(_SECTION_TITLES["imptors"])
        for k, v in enumerate(prmdict["imptors"][1]):
            line = f"{'imptors':12s}{_c(prmdict['imptors'][0][k], 4):<25s} "
            for i in range(0, len(v), 3):
                line += f"{v[i]:7.3f} {v[i + 1]:2.1f} {int(v[i + 2]):d}"
            out += line + "\n"

    if len(prmdict["bndcflux"][0]) > 0:
        out += _hdr(_SECTION_TITLES["cflux"])
        for k, v in enumerate(prmdict["bndcflux"][1]):
            out += f"{'bndcflux':12s}{_c(prmdict['bndcflux'][0][k], 2):<16s}{v:10.6f}\n"
    if len(prmdict["angcflux"][0]) > 0:
        for k, v in enumerate(prmdict["angcflux"][1]):
            c = _c(prmdict["angcflux"][0][k], 3)
            out += f"{'angcflux':12s}{c:<16s}{v[0]:10.6f} {v[1]:10.6f} {v[2]:10.6f} {v[3]:10.6f}\n"

    if len(prmdict["exchpol"]) > 0:
        out += _hdr(_SECTION_TITLES["exchpol"])
        for vals in prmdict["exchpol"]:
            t = typcls[int(vals[0])]
            out += (f"{'exchpol':16s} {t:<11d}{vals[1]:10.6f} {vals[2]:10.6f} "
                    f"{vals[3]:10.6f} {int(vals[-1]):<10d}\n")

    Path(fnout).write_text(out)
    return out


def _c(ids, n: int) -> str:
    """Format n class/type ids as the legacy '%3d  %3d ...' column block."""
    ids = [int(a) for a in ids]
    return "  ".join(f"{ids[i]:3d}" for i in range(n))


def write_key(
    prm_filename: str,
    fnout: str | Path,
    *,
    opt: str = "liquid",
    a_axis: float = 30.0,
    extra_lines: list[str] | None = None,
) -> str:
    """Write a Tinker ``.key`` referencing ``prm_filename``.

    ``opt='liquid'`` uses the NPT/RESPA template; ``opt='gas'`` the stochastic
    gas-phase template. ``extra_lines`` are appended verbatim.
    """
    template = KEY_GAS if opt == "gas" else KEY_LIQUID
    text = template.format(prm=prm_filename, a_axis=a_axis)
    if extra_lines:
        text += "\n".join(extra_lines) + "\n"
    Path(fnout).write_text(text)
    return text


__all__ = [
    "process_prm",
    "write_prm",
    "write_key",
    "multipole_factors",
    "compute_total_charge",
    "PRM_HEADER",
    "KEY_LIQUID",
    "KEY_GAS",
]

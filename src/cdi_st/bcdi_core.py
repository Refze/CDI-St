"""bcdi_core.py v6.0 — dislocation loops, CIF import, rect detector, detector presets, export."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from diffpy.structure import Atom, Lattice, Structure
from pymatgen.core.lattice import Lattice as PmgLattice
from pymatgen.core.structure import Structure as PmgStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    import xrayutilities as xu

    _HAS_XU = True
except:
    xu = None
    _HAS_XU = False
warnings.filterwarnings("ignore")


def _6c_R3m(z):
    return [
        [0, 0, z % 1],
        [2 / 3, 1 / 3, (z + 1 / 3) % 1],
        [1 / 3, 2 / 3, (z + 2 / 3) % 1],
        [0, 0, (1 - z) % 1],
        [2 / 3, 1 / 3, (1 - z + 1 / 3) % 1],
        [1 / 3, 2 / 3, (1 - z + 2 / 3) % 1],
    ]


def _3a_R3m():
    return [[0, 0, 0], [2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]]


def _3_from_6c_R3m(z):
    return [
        [0, 0, z % 1],
        [2 / 3, 1 / 3, (z + 1 / 3) % 1],
        [1 / 3, 2 / 3, (z + 2 / 3) % 1],
    ]


DETECTOR_PRESETS = {
    "Custom": {"nx": 128, "ny": 128, "pixel_um": 55.0},
    "Eiger2 500K": {"nx": 1028, "ny": 512, "pixel_um": 75.0},
    "Eiger2 1M": {"nx": 1028, "ny": 1062, "pixel_um": 75.0},
    "Eiger2 4M": {"nx": 2068, "ny": 2162, "pixel_um": 75.0},
    "Maxipix 1x1": {"nx": 256, "ny": 256, "pixel_um": 55.0},
    "Maxipix 2x2": {"nx": 516, "ny": 516, "pixel_um": 55.0},
    "Merlin Quad": {"nx": 512, "ny": 512, "pixel_um": 55.0},
    "Pilatus 100K": {"nx": 487, "ny": 195, "pixel_um": 172.0},
    "Pilatus 300K": {"nx": 487, "ny": 619, "pixel_um": 172.0},
    "Timepix3": {"nx": 256, "ny": 256, "pixel_um": 55.0},
}

MATERIAL_PRESETS = {
    "Al": {
        "crystal_type": "fcc",
        "a": 4.046,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["Al"],
        "basis": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        "species_per_site": ["Al"] * 4,
        "space_group": "Fm-3m",
        "xu_builtin": "Al",
        "B_iso": {"Al": 0.77},
        "formula": "Al",
    },
    "Cu": {
        "crystal_type": "fcc",
        "a": 3.615,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["Cu"],
        "basis": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        "species_per_site": ["Cu"] * 4,
        "space_group": "Fm-3m",
        "xu_builtin": "Cu",
        "B_iso": {"Cu": 0.86},
        "formula": "Cu",
    },
    "Au": {
        "crystal_type": "fcc",
        "a": 4.078,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["Au"],
        "basis": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        "species_per_site": ["Au"] * 4,
        "space_group": "Fm-3m",
        "xu_builtin": "Au",
        "B_iso": {"Au": 0.69},
        "formula": "Au",
    },
    "W": {
        "crystal_type": "bcc",
        "a": 3.165,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["W"],
        "basis": [[0, 0, 0], [0.5, 0.5, 0.5]],
        "species_per_site": ["W", "W"],
        "space_group": "Im-3m",
        "xu_builtin": "W",
        "B_iso": {"W": 0.28},
        "formula": "W",
    },
    "Fe": {
        "crystal_type": "bcc",
        "a": 2.867,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["Fe"],
        "basis": [[0, 0, 0], [0.5, 0.5, 0.5]],
        "species_per_site": ["Fe", "Fe"],
        "space_group": "Im-3m",
        "xu_builtin": "Fe",
        "B_iso": {"Fe": 0.35},
        "formula": "Fe",
    },
    "Si": {
        "crystal_type": "diamond",
        "a": 5.431,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["Si"],
        "basis": [
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0],
            [0.75, 0.75, 0.25],
            [0.5, 0, 0.5],
            [0.75, 0.25, 0.75],
            [0, 0.5, 0.5],
            [0.25, 0.75, 0.75],
        ],
        "species_per_site": ["Si"] * 8,
        "space_group": "Fd-3m",
        "xu_builtin": "Si",
        "B_iso": {"Si": 0.47},
        "formula": "Si",
    },
    "Ge": {
        "crystal_type": "diamond",
        "a": 5.658,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["Ge"],
        "basis": [
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0],
            [0.75, 0.75, 0.25],
            [0.5, 0, 0.5],
            [0.75, 0.25, 0.75],
            [0, 0.5, 0.5],
            [0.25, 0.75, 0.75],
        ],
        "species_per_site": ["Ge"] * 8,
        "space_group": "Fd-3m",
        "xu_builtin": None,
        "B_iso": {"Ge": 0.57},
        "formula": "Ge",
    },
    "Pt": {
        "crystal_type": "fcc",
        "a": 3.924,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["Pt"],
        "basis": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        "species_per_site": ["Pt"] * 4,
        "space_group": "Fm-3m",
        "xu_builtin": "Pt",
        "B_iso": {"Pt": 0.50},
        "formula": "Pt",
    },
    "SiC_3C": {
        "crystal_type": "custom",
        "a": 4.359,
        "b": None,
        "c": None,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "species": ["Si", "C"],
        "basis": [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        "species_per_site": ["Si"] * 4 + ["C"] * 4,
        "space_group": "F-43m",
        "xu_builtin": None,
        "B_iso": {"Si": 0.35, "C": 0.30},
        "formula": "3C-SiC",
    },
    "SiC_4H": {
        "crystal_type": "custom",
        "a": 3.073,
        "b": None,
        "c": 10.053,
        "alpha": 90,
        "beta": 90,
        "gamma": 120,
        "species": ["Si", "C"],
        "basis": [
            [0, 0, 0],
            [1 / 3, 2 / 3, 0.25],
            [0, 0, 0.5],
            [1 / 3, 2 / 3, 0.75],
            [0, 0, 0.1875],
            [1 / 3, 2 / 3, 0.4375],
            [0, 0, 0.6875],
            [1 / 3, 2 / 3, 0.9375],
        ],
        "species_per_site": ["Si"] * 4 + ["C"] * 4,
        "space_group": "P6_3mc",
        "xu_builtin": None,
        "B_iso": {"Si": 0.35, "C": 0.30},
        "formula": "4H-SiC",
    },
    "ZnO": {
        "crystal_type": "wurtzite",
        "a": 3.250,
        "b": None,
        "c": 5.207,
        "alpha": 90,
        "beta": 90,
        "gamma": 120,
        "species": ["Zn", "O"],
        "basis": [
            [1 / 3, 2 / 3, 0],
            [1 / 3, 2 / 3, 0.3825],
            [2 / 3, 1 / 3, 0.5],
            [2 / 3, 1 / 3, 0.8825],
        ],
        "species_per_site": ["Zn", "O", "Zn", "O"],
        "space_group": "P6_3mc",
        "xu_builtin": None,
        "B_iso": {"Zn": 0.56, "O": 0.70},
        "formula": "ZnO",
    },
    "GaN": {
        "crystal_type": "wurtzite",
        "a": 3.189,
        "b": None,
        "c": 5.185,
        "alpha": 90,
        "beta": 90,
        "gamma": 120,
        "species": ["Ga", "N"],
        "basis": [
            [1 / 3, 2 / 3, 0],
            [1 / 3, 2 / 3, 0.377],
            [2 / 3, 1 / 3, 0.5],
            [2 / 3, 1 / 3, 0.877],
        ],
        "species_per_site": ["Ga", "N", "Ga", "N"],
        "space_group": "P6_3mc",
        "xu_builtin": None,
        "B_iso": {"Ga": 0.39, "N": 0.55},
        "formula": "GaN",
    },
    "Te": {
        "crystal_type": "custom",
        "a": 4.458,
        "b": None,
        "c": 5.927,
        "alpha": 90,
        "beta": 90,
        "gamma": 120,
        "species": ["Te"],
        "basis": [[0.2636, 0, 1 / 3], [0, 0.2636, 2 / 3], [0.7364, 0.7364, 0]],
        "species_per_site": ["Te"] * 3,
        "space_group": "P3_121",
        "xu_builtin": None,
        "B_iso": {"Te": 1.20},
        "formula": "Te",
    },
    "Fe3GeTe2": {
        "crystal_type": "custom",
        "a": 3.991,
        "b": None,
        "c": 16.333,
        "alpha": 90,
        "beta": 90,
        "gamma": 120,
        "species": ["Fe", "Ge", "Te"],
        "basis": [
            [0, 0, 0.6718],
            [0, 0, 0.3282],
            [0, 0, 0.1718],
            [0, 0, 0.8282],
            [1 / 3, 2 / 3, 0.25],
            [2 / 3, 1 / 3, 0.75],
            [1 / 3, 2 / 3, 0.75],
            [2 / 3, 1 / 3, 0.25],
            [1 / 3, 2 / 3, 0.40982],
            [2 / 3, 1 / 3, 0.90982],
            [2 / 3, 1 / 3, 0.59018],
            [1 / 3, 2 / 3, 0.09018],
        ],
        "species_per_site": ["Fe"] * 6 + ["Ge"] * 2 + ["Te"] * 4,
        "space_group": "P6_3/mmc",
        "xu_builtin": None,
        "B_iso": {"Fe": 0.40, "Ge": 0.55, "Te": 1.0},
        "formula": "Fe3GeTe2",
    },
    "Fe5GeTe2": {
        "crystal_type": "custom",
        "a": 4.043,
        "b": None,
        "c": 29.190,
        "alpha": 90,
        "beta": 90,
        "gamma": 120,
        "species": ["Fe", "Ge", "Te"],
        "basis": _6c_R3m(0.0327)
        + _3a_R3m()
        + _6c_R3m(0.0991)
        + _3_from_6c_R3m(0.0654)
        + _3_from_6c_R3m(0.1977)
        + _3_from_6c_R3m(0.1313),
        "species_per_site": ["Fe"] * 15 + ["Ge"] * 3 + ["Te"] * 6,
        "space_group": "R3m",
        "xu_builtin": None,
        "B_iso": {"Fe": 0.45, "Ge": 0.60, "Te": 1.10},
        "formula": "Fe5GeTe2",
    },
}


def load_cif_as_preset(cif_path):
    """Load a CIF file and return a material dict compatible with MATERIAL_PRESETS."""
    s = PmgStructure.from_file(cif_path)
    sga = SpacegroupAnalyzer(s, symprec=0.1)
    sg_sym = sga.get_space_group_symbol()
    formula = s.composition.reduced_formula
    basis = []
    species = []
    for site in s:
        basis.append(list(site.frac_coords % 1.0))
        species.append(site.species_string)
    unique_sp = list(dict.fromkeys(species))
    return {
        "crystal_type": "custom",
        "a": round(s.lattice.a, 4),
        "b": round(s.lattice.b, 4),
        "c": round(s.lattice.c, 4),
        "alpha": round(s.lattice.alpha, 2),
        "beta": round(s.lattice.beta, 2),
        "gamma": round(s.lattice.gamma, 2),
        "species": unique_sp,
        "basis": basis,
        "species_per_site": species,
        "space_group": sg_sym,
        "xu_builtin": None,
        "B_iso": {sp: 0.5 for sp in unique_sp},
        "formula": formula,
    }


def default_shape_for_material(n):
    if n not in MATERIAL_PRESETS:
        return "cube"
    m = MATERIAL_PRESETS[n]
    ct = m["crystal_type"]
    if ct in ("fcc", "bcc", "diamond"):
        return "cube"
    if ct in ("hcp", "wurtzite") or m["gamma"] == 120:
        return "hexagonal"
    return "cube"


def compatible_shapes(n):
    if n not in MATERIAL_PRESETS:
        return ["cube", "rectangle", "cylinder", "hexagonal", "sphere"]
    m = MATERIAL_PRESETS[n]
    ct = m["crystal_type"]
    if ct in ("fcc", "bcc", "diamond"):
        return ["cube", "rectangle", "cylinder", "sphere"]
    if ct in ("hcp", "wurtzite") or m["gamma"] == 120:
        return ["hexagonal", "cylinder", "sphere"]
    return ["cube", "rectangle", "cylinder", "hexagonal", "sphere"]


def find_preset(q):
    q = q.strip()
    if q in MATERIAL_PRESETS:
        return q
    for n, p in MATERIAL_PRESETS.items():
        if p.get("formula", "").lower() == q.lower() or n.lower() == q.lower():
            return n
    return None


class DislocationConfig:
    def __init__(
        self,
        dtype="edge",
        pos_frac=(0.5, 0.5),
        line_dir="Z",
        b_angstrom=None,
        nu=0.3,
        b_edge=None,
        b_screw=None,
    ):
        self.dtype = dtype
        self.pos_frac = pos_frac
        self.line_dir = line_dir
        self.b_angstrom = b_angstrom
        self.nu = nu
        self.b_edge = b_edge
        self.b_screw = b_screw


class DislocationLoopConfig:
    """Prismatic dislocation loop: atomsk-style parameters."""

    def __init__(
        self,
        center_frac=(0.5, 0.5, 0.5),
        radius_angstrom=20.0,
        b_angstrom=None,
        normal="Z",
        nu=0.3,
    ):
        self.center_frac = center_frac
        self.radius_angstrom = radius_angstrom
        self.b_angstrom = b_angstrom
        self.normal = normal
        self.nu = nu


class BCDIConfig:
    HC_KEV_ANG = 12.3984

    def __init__(self, material_name="Al"):
        if material_name not in MATERIAL_PRESETS:
            raise ValueError(f"Unknown '{material_name}'")
        mat = MATERIAL_PRESETS[material_name]
        self.MATERIAL_NAME = material_name
        self.CRYSTAL_TYPE = mat["crystal_type"]
        self.SPACE_GROUP = mat["space_group"]
        self.LATTICE_A = mat["a"]
        self.LATTICE_B = mat.get("b") or mat["a"]
        self.LATTICE_C = mat.get("c") or mat["a"]
        self.LATTICE_ALPHA = mat["alpha"]
        self.LATTICE_BETA = mat["beta"]
        self.LATTICE_GAMMA = mat["gamma"]
        self.ATOMIC_SPECIES = mat["species"]
        self.BASIS_COORDS = [[f % 1.0 for f in pos] for pos in mat["basis"]]
        self.SPECIES_PER_SITE = mat.get(
            "species_per_site", mat["species"] * len(mat["basis"])
        )
        self.XU_BUILTIN = mat.get("xu_builtin")
        self.B_ISO = mat.get("B_iso", {})
        self.BEAM_ENERGY_KEV = 10.0
        self.POLARIZATION_FACTOR = 1.0
        self.BEAM_SIZE_UM = 1.0
        self.SUPERCELL_MULT = (20, 20, 20)
        self.PARTICLE_SHAPE = default_shape_for_material(material_name)
        self.DISLOCATION = None
        self.DISLOCATION_LOOP = None
        self.DETECTOR_PIXEL_UM = 55.0
        self.SAMPLE_DETECTOR_DISTANCE_M = 0.5
        self.DETECTOR_NX = 128
        self.DETECTOR_NY = 128
        self.ROCKING_STEPS = 128
        self.ROCKING_STEP_DEG = 0.00315
        self.SOURCE_SIZE_H_UM = 300.0
        self.SOURCE_SIZE_V_UM = 10.0
        self.SOURCE_DISTANCE_M = 50.0
        self.MONOCHR_BANDWIDTH = 1e-4
        self.STRAIN_TYPE = "none"
        self.STRAIN_MAGNITUDE = 1e-4
        self.TARGET_OVERSAMPLING = 5.0
        self.ANGULAR_OFFSET_DEG = 0.0
        self.HKL_MAX_SEARCH = 5
        self.Q_MAX_INV_ANG = 8.0
        self.TWO_THETA_MIN = 10.0
        self.TWO_THETA_MAX = 90.0
        self.OUTPUT_DIR = Path("bcdi_output")
        self.OUTPUT_DIR.mkdir(exist_ok=True)

    @property
    def DETECTOR_N_PIXELS(self):
        return max(self.DETECTOR_NX, self.DETECTOR_NY)

    @property
    def wavelength_angstrom(self):
        return self.HC_KEV_ANG / self.BEAM_ENERGY_KEV

    @property
    def lattice_vectors_cartesian(self):
        a, b, c = self.LATTICE_A, self.LATTICE_B, self.LATTICE_C
        al, be, ga = [
            np.radians(x)
            for x in (self.LATTICE_ALPHA, self.LATTICE_BETA, self.LATTICE_GAMMA)
        ]
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
        cx = c * np.cos(be)
        cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        cz = np.sqrt(max(0.0, c**2 - cx**2 - cy**2))
        return np.array([a1, a2, np.array([cx, cy, cz])])

    @property
    def reciprocal_lattice_vectors(self):
        M = self.lattice_vectors_cartesian
        V = np.dot(M[0], np.cross(M[1], M[2]))
        return np.array(
            [
                2 * np.pi * np.cross(M[1], M[2]) / V,
                2 * np.pi * np.cross(M[2], M[0]) / V,
                2 * np.pi * np.cross(M[0], M[1]) / V,
            ]
        )

    @property
    def particle_size_angstrom(self):
        M = self.lattice_vectors_cartesian
        nx, ny, nz = self.SUPERCELL_MULT
        sc = np.array([nx * M[0], ny * M[1], nz * M[2]])
        c = np.array(
            [
                i * sc[0] + j * sc[1] + k * sc[2]
                for i in (0, 1)
                for j in (0, 1)
                for k in (0, 1)
            ]
        )
        return c.max(0) - c.min(0)

    @property
    def particle_size_nm(self):
        return self.particle_size_angstrom / 10.0

    @property
    def unit_cell_volume_ang3(self):
        M = self.lattice_vectors_cartesian
        return abs(np.dot(M[0], np.cross(M[1], M[2])))


class CrystalBuilder:
    def __init__(self, config):
        self.config = config
        self.unit_cell = None
        self.pmg_structure = None
        self.supercell_positions_ang = None
        self.supercell_species = None
        self._cell_indices = None
        self._displacement_magnitudes = None

    def build(self, progress_cb=None):
        cfg = self.config
        lat = Lattice(
            cfg.LATTICE_A,
            cfg.LATTICE_B,
            cfg.LATTICE_C,
            cfg.LATTICE_ALPHA,
            cfg.LATTICE_BETA,
            cfg.LATTICE_GAMMA,
        )
        self.unit_cell = Structure(
            atoms=[
                Atom(cfg.SPECIES_PER_SITE[i], p) for i, p in enumerate(cfg.BASIS_COORDS)
            ],
            lattice=lat,
        )
        self.pmg_structure = PmgStructure(
            PmgLattice.from_parameters(
                cfg.LATTICE_A,
                cfg.LATTICE_B,
                cfg.LATTICE_C,
                cfg.LATTICE_ALPHA,
                cfg.LATTICE_BETA,
                cfg.LATTICE_GAMMA,
            ),
            cfg.SPECIES_PER_SITE,
            cfg.BASIS_COORDS,
        )
        self._build_sc(progress_cb)
        return True

    def _build_sc(self, cb=None):
        cfg = self.config
        nx, ny, nz = cfg.SUPERCELL_MULT
        M = cfg.lattice_vectors_cartesian
        nb = len(cfg.BASIS_COORDS)
        bc = np.array(
            [
                np.array(f)[0] * M[0] + np.array(f)[1] * M[1] + np.array(f)[2] * M[2]
                for f in cfg.BASIS_COORDS
            ]
        )
        tot = nx * ny * nz
        pos = np.empty((tot * nb, 3), np.float64)
        sp = []
        ci = []
        k = 0
        d = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    off = ix * M[0] + iy * M[1] + iz * M[2]
                    for ib in range(nb):
                        pos[k] = off + bc[ib]
                        sp.append(cfg.SPECIES_PER_SITE[ib % len(cfg.SPECIES_PER_SITE)])
                        ci.append((ix, iy, iz))
                        k += 1
                    d += 1
                    if cb and d % max(1, tot // 50) == 0:
                        cb(int(100 * d / tot))
        self.supercell_positions_ang = pos[:k]
        self.supercell_species = sp
        self._cell_indices = ci
        if cb:
            cb(100)

    def apply_shape_filter(self):
        cfg = self.config
        s = cfg.PARTICLE_SHAPE
        if s in ("cube", "rectangle") or self.supercell_positions_ang is None:
            return
        p = self.supercell_positions_ang
        M = cfg.lattice_vectors_cartesian
        nx, ny, nz = cfg.SUPERCELL_MULT
        center = 0.5 * (nx * M[0] + ny * M[1] + nz * M[2])
        r = p - center
        Lx = np.linalg.norm(nx * M[0])
        Ly = np.linalg.norm(ny * M[1])
        Lz = np.linalg.norm(nz * M[2])
        R_xy = 0.47 * min(Lx, Ly)
        if s == "cylinder":
            mask = (r[:, 0] ** 2 + r[:, 1] ** 2) <= R_xy**2
        elif s == "sphere":
            R = 0.47 * min(Lx, Ly, Lz)
            mask = (r**2).sum(1) <= R**2
        elif s == "hexagonal":
            s3 = np.sqrt(3.0) / 2.0
            mask = (
                np.maximum(
                    np.abs(r[:, 0]),
                    np.maximum(
                        np.abs(0.5 * r[:, 0] + s3 * r[:, 1]),
                        np.abs(0.5 * r[:, 0] - s3 * r[:, 1]),
                    ),
                )
                <= R_xy
            )
        else:
            return
        self.supercell_positions_ang = p[mask]
        self.supercell_species = [
            sv for sv, m in zip(self.supercell_species, mask) if m
        ]
        if self._cell_indices:
            self._cell_indices = [c for c, m in zip(self._cell_indices, mask) if m]

    def apply_dislocation_displacement(self):
        dc = self.config.DISLOCATION
        if dc is None or self.supercell_positions_ang is None:
            return
        pos_ideal = self.supercell_positions_ang.copy()
        pos = pos_ideal.copy()
        center = (pos.max(0) + pos.min(0)) / 2.0
        ext = pos.max(0) - pos.min(0) + 1e-10
        b = dc.b_angstrom if dc.b_angstrom else self.config.LATTICE_A
        nu = dc.nu
        px = center[0] + (dc.pos_frac[0] - 0.5) * ext[0]
        py = center[1] + (dc.pos_frac[1] - 0.5) * ext[1]
        if dc.line_dir == "Z":
            DX = pos[:, 0] - px
            DY = pos[:, 1] - py
            di = (0, 1)
        elif dc.line_dir == "Y":
            DX = pos[:, 0] - px
            DY = pos[:, 2] - py
            di = (0, 2)
        else:
            DX = pos[:, 1] - px
            DY = pos[:, 2] - py
            di = (1, 2)
        R2 = DX**2 + DY**2 + 1e-20
        theta = np.arctan2(DY, DX)
        if dc.dtype == "edge":
            pos[:, di[0]] += (b / (2 * np.pi)) * (theta + DX * DY / (2 * (1 - nu) * R2))
        elif dc.dtype == "screw":
            la = {"Z": 2, "Y": 1, "X": 0}[dc.line_dir]
            pos[:, la] += (b / (2 * np.pi)) * theta
        elif dc.dtype == "mixed":
            be = dc.b_edge if dc.b_edge else b * 0.866
            bs = dc.b_screw if dc.b_screw else b * 0.5
            pos[:, di[0]] += (be / (2 * np.pi)) * (
                theta + DX * DY / (2 * (1 - nu) * R2)
            )
            la = {"Z": 2, "Y": 1, "X": 0}[dc.line_dir]
            pos[:, la] += (bs / (2 * np.pi)) * theta
        self.supercell_positions_ang = pos
        self._displacement_magnitudes = np.linalg.norm(pos - pos_ideal, axis=1)

    def apply_dislocation_loop(self):
        """Prismatic dislocation loop: u(r) = (b/4pi)*Omega(r) where Omega is solid angle."""
        dl = self.config.DISLOCATION_LOOP
        if dl is None or self.supercell_positions_ang is None:
            return
        pos_ideal = self.supercell_positions_ang.copy()
        pos = pos_ideal.copy()
        center = (pos.max(0) + pos.min(0)) / 2.0
        ext = pos.max(0) - pos.min(0) + 1e-10
        b = dl.b_angstrom if dl.b_angstrom else self.config.LATTICE_A
        R = dl.radius_angstrom
        cx = center[0] + (dl.center_frac[0] - 0.5) * ext[0]
        cy = center[1] + (dl.center_frac[1] - 0.5) * ext[1]
        cz = center[2] + (dl.center_frac[2] - 0.5) * ext[2]
        ndir = {
            "X": np.array([1, 0, 0]),
            "Y": np.array([0, 1, 0]),
            "Z": np.array([0, 0, 1]),
        }[dl.normal]
        # Vector from loop center to each atom
        dx = pos[:, 0] - cx
        dy = pos[:, 1] - cy
        dz = pos[:, 2] - cz
        d_along = dx * ndir[0] + dy * ndir[1] + dz * ndir[2]
        perp_x = dx - d_along * ndir[0]
        perp_y = dy - d_along * ndir[1]
        perp_z = dz - d_along * ndir[2]
        rho = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
        # Solid angle: Omega = 2*pi*sign(d)*(1 - |d|/sqrt(d^2+R^2)) for rho<R (inside)
        #              Omega ~ pi*R^2*d/(d^2+rho^2)^(3/2) for rho>R (outside)
        dist2 = d_along**2 + 1e-20
        inside = rho < R
        Omega = np.zeros(len(pos))
        # Inside the loop projection
        Omega[inside] = (
            2
            * np.pi
            * np.sign(d_along[inside] + 1e-30)
            * (1 - np.abs(d_along[inside]) / np.sqrt(dist2[inside] + R**2))
        )
        # Outside: approximate
        r3 = (dist2[~inside] + rho[~inside] ** 2) ** 1.5 + 1e-30
        Omega[~inside] = np.pi * R**2 * d_along[~inside] / r3
        # Displacement: u = b/(4*pi) * Omega along normal
        u_mag = b / (4 * np.pi) * Omega
        pos[:, 0] += u_mag * ndir[0]
        pos[:, 1] += u_mag * ndir[1]
        pos[:, 2] += u_mag * ndir[2]
        self.supercell_positions_ang = pos
        self._displacement_magnitudes = np.linalg.norm(pos - pos_ideal, axis=1)

    def get_pmg_structure(self):
        if self.pmg_structure is None:
            self.build()
        return self.pmg_structure

    def get_lattice_atoms(self, max_atoms=50000):
        if self.supercell_positions_ang is None:
            return None, None, None
        p = self.supercell_positions_ang
        s = self.supercell_species
        n = len(p)
        disp = self._displacement_magnitudes
        if n <= max_atoms:
            return p, s, disp
        cfg = self.config
        nx, ny, nz = cfg.SUPERCELL_MULT
        nb = len(cfg.BASIS_COORDS)
        n_cells_target = max(1, max_atoms // nb)
        ratio = (nx * ny * nz / n_cells_target) ** (1.0 / 3.0)
        sx = max(1, int(round(ratio)))
        sy = max(1, int(round(ratio)))
        sz = max(1, int(round(ratio)))
        if nx < ny * 0.3:
            sx = 1
        if ny < nx * 0.3:
            sy = 1
        if nz < nx * 0.3 or nz < ny * 0.3:
            sz = 1
        if self._cell_indices:
            keep = set()
            for ix in range(0, nx, sx):
                for iy in range(0, ny, sy):
                    for iz in range(0, nz, sz):
                        keep.add((ix, iy, iz))
            mask = np.array([c in keep for c in self._cell_indices])
            dp = disp[mask] if disp is not None else None
            return p[mask], [s[i] for i, m in enumerate(mask) if m], dp
        st = max(1, n // max_atoms)
        idx = np.arange(0, n, st)
        dp = disp[idx] if disp is not None else None
        return p[idx], [s[i] for i in idx], dp


class ReflectionCalculator:
    def __init__(self, config, builder):
        self.config = config
        self.builder = builder
        self.crystal_xu = None
        self.reflections_df = None
        if _HAS_XU and config.XU_BUILTIN:
            try:
                self.crystal_xu = getattr(xu.materials, config.XU_BUILTIN)
            except:
                pass

    def _d(self, h, k, l):
        c = self.config
        a, b, cc = c.LATTICE_A, c.LATTICE_B, c.LATTICE_C
        V = c.unit_cell_volume_ang3
        al, be, ga = [
            np.radians(x) for x in (c.LATTICE_ALPHA, c.LATTICE_BETA, c.LATTICE_GAMMA)
        ]
        ca, cb, cg = np.cos(al), np.cos(be), np.cos(ga)
        sa, sb, sg = np.sin(al), np.sin(be), np.sin(ga)
        id2 = (1.0 / V**2) * (
            h**2 * b**2 * cc**2 * sa**2
            + k**2 * a**2 * cc**2 * sb**2
            + l**2 * a**2 * b**2 * sg**2
            + 2 * h * k * a * b * cc**2 * (ca * cb - cg)
            + 2 * k * l * a**2 * b * cc * (cb * cg - ca)
            + 2 * h * l * a * b**2 * cc * (ca * cg - cb)
        )
        return 1.0 / np.sqrt(id2) if id2 > 0 else None

    def _F(self, h, k, l, eV, stol):
        c = self.config
        F = 0j
        for i, pos in enumerate(c.BASIS_COORDS):
            sp = c.SPECIES_PER_SITE[i]
            from pymatgen.core.periodic_table import Element as PE

            try:
                Z = PE(sp).Z
            except:
                Z = 1
            f0 = float(Z)
            if _HAS_XU:
                try:
                    v = xu.math.fatomic(Z, stol)
                    f0 = (
                        float(v[0])
                        if hasattr(v, "__len__") and len(v) > 0
                        else float(v)
                    )
                except:
                    pass
                try:
                    fp, fpp = xu.math.fatomic_dispersion(sp, eV)
                    if hasattr(fp, "__len__"):
                        fp, fpp = float(fp[0]), float(fpp[0])
                    ft = f0 + fp + 1j * fpp
                except:
                    ft = f0 + 0j
            else:
                ft = f0 + 0j
            ft *= np.exp(-c.B_ISO.get(sp, 0.0) * stol**2)
            F += ft * np.exp(1j * 2 * np.pi * (h * pos[0] + k * pos[1] + l * pos[2]))
        return F

    def _lp(self, tr):
        P = self.config.POLARIZATION_FACTOR
        tt = 2.0 * tr
        return (P * np.cos(tt) ** 2 + (1.0 - P)) / (
            np.sin(tr) ** 2 * np.cos(tr) + 1e-12
        )

    def _fk(self, hkl, ops):
        eq = set()
        for op in ops:
            he = tuple(int(round(x)) for x in op.rotation_matrix @ hkl)
            eq.add(he)
            eq.add(tuple(-x for x in he))
        return frozenset(eq)

    def calculate(self):
        c = self.config
        lam = c.wavelength_angstrom
        eV = c.BEAM_ENERGY_KEV * 1e3
        mh = c.HKL_MAX_SEARCH
        rows = []
        try:
            sga = SpacegroupAnalyzer(self.builder.get_pmg_structure())
            ops = sga.get_symmetry_operations()
        except:
            ops = None
        seen = set()
        for h in range(-mh, mh + 1):
            for k in range(-mh, mh + 1):
                for l in range(-mh, mh + 1):
                    if h == k == l == 0:
                        continue
                    d = self._d(h, k, l)
                    if d is None or d <= 0:
                        continue
                    sa = lam / (2.0 * d)
                    if abs(sa) >= 1.0:
                        continue
                    tt = 2.0 * np.degrees(np.arcsin(sa))
                    tr = np.radians(tt / 2.0)
                    if not (c.TWO_THETA_MIN <= tt <= c.TWO_THETA_MAX):
                        continue
                    q = 4.0 * np.pi * np.sin(tr) / lam
                    if q > c.Q_MAX_INV_ANG:
                        continue
                    stol = np.sin(tr) / lam
                    F = self._F(h, k, l, eV, stol)
                    Fa = np.abs(F)
                    F2 = Fa**2
                    if F2 < 1e-4:
                        continue
                    LP = self._lp(tr)
                    if ops is not None:
                        fk = self._fk(np.array([h, k, l]), ops)
                        iu = fk not in seen
                        if iu:
                            seen.add(fk)
                    else:
                        iu = (h >= 0) and (h >= k) and (k >= l) and (l >= 0)
                    rows.append(
                        {
                            "hkl": (h, k, l),
                            "hkl_str": f"({h} {k} {l})",
                            "d_Ang": round(d, 4),
                            "2theta": round(tt, 4),
                            "theta_B": round(tt / 2.0, 4),
                            "q_Ang": round(q, 4),
                            "|F|": round(Fa, 3),
                            "F_sq": round(F2, 2),
                            "LP": round(LP, 4),
                            "unique": iu,
                        }
                    )
        df = pd.DataFrame(rows).sort_values("q_Ang").reset_index(drop=True)
        df = df[df["unique"]].copy().reset_index(drop=True)
        df["I_rel"] = df["F_sq"] * df["LP"]
        mx = df["I_rel"].max() if len(df) else 1.0
        df["I_norm"] = (df["I_rel"] / mx) if mx > 0 else 0.0
        df = df.sort_values("I_norm", ascending=False).reset_index(drop=True)
        df["BCDI_flag"] = df["I_norm"] >= 0.3

        def _can(t):
            h, k, l = t
            n = (-h, -k, -l)
            s1 = sum(t)
            s2 = sum(n)
            if s2 > s1:
                return n
            if s1 > s2:
                return t
            p1 = sum(1 for x in t if x > 0)
            p2 = sum(1 for x in n if x > 0)
            if p2 > p1:
                return n
            if p1 > p2:
                return t
            return t if t > n else n

        df["hkl_canonical"] = df["hkl"].apply(_can)
        df["hkl_display"] = df["hkl_canonical"].apply(
            lambda t: f"({t[0]} {t[1]} {t[2]})"
        )
        self.reflections_df = df
        return df

    def select_reflection(self, hkl=None):
        if self.reflections_df is None:
            self.calculate()
        df = self.reflections_df
        if hkl is None:
            cands = df[df["BCDI_flag"]]
            sel = cands.iloc[0] if len(cands) else df.iloc[0]
        else:
            h, k, l = hkl
            m = pd.DataFrame()
            for th, tk, tl in [(h, k, l), (-h, -k, -l)]:
                ts = f"({th} {tk} {tl})"
                m = df[df["hkl_str"] == ts]
                if len(m):
                    break
            if len(m) == 0:
                for th, tk, tl in [(h, k, l), (-h, -k, -l)]:
                    ts = f"({th} {tk} {tl})"
                    m = df[df["hkl_display"] == ts]
                    if len(m):
                        break
            if len(m) == 0:
                raise ValueError(f"Reflection ({h} {k} {l}) not found")
            sel = m.iloc[0]
        return {
            "hkl": sel["hkl"],
            "hkl_str": sel["hkl_str"],
            "d_hkl": sel["d_Ang"],
            "theta_B_rad": np.radians(sel["theta_B"]),
            "two_theta_rad": np.radians(sel["2theta"]),
            "q_magnitude": sel["q_Ang"],
            "F_squared": sel["F_sq"],
            "LP_factor": sel["LP"],
            "intensity_norm": sel["I_norm"],
        }


class BCDISimulator:
    def __init__(self, config, reflection, crystal_builder=None):
        self.config = config
        self.reflection = reflection
        self.builder = crystal_builder
        self.diff_volume = None
        self.q_grids = None
        self.coherence = {}

    def _coh(self):
        c = self.config
        lam = c.wavelength_angstrom
        RA = c.SOURCE_DISTANCE_M * 1e10
        sh = c.SOURCE_SIZE_H_UM * 1e4
        sv = c.SOURCE_SIZE_V_UM * 1e4
        return (
            lam * RA / (2 * np.pi * sh) if sh > 0 else np.inf,
            lam * RA / (2 * np.pi * sv) if sv > 0 else np.inf,
            lam / (2 * c.MONOCHR_BANDWIDTH) if c.MONOCHR_BANDWIDTH > 0 else np.inf,
        )

    def _pc(self, vol, dx, dy, dz):
        from scipy.ndimage import gaussian_filter

        c = self.config
        k0 = 2 * np.pi / c.wavelength_angstrom
        RA = c.SOURCE_DISTANCE_M * 1e10
        sx = k0 * (c.SOURCE_SIZE_H_UM * 1e4) / RA / dx if dx > 0 and RA > 0 else 0.0
        sy = k0 * (c.SOURCE_SIZE_V_UM * 1e4) / RA / dy if dy > 0 and RA > 0 else 0.0
        sz = k0 * c.MONOCHR_BANDWIDTH / dz if dz > 0 else 0.0
        if max(sx, sy, sz) < 0.1:
            return vol
        return gaussian_filter(vol.astype(np.float64), sigma=[sx, sy, sz]).astype(
            np.float32
        )

    def _displacement(self, N, tA, OS):
        cfg = self.config
        G = self.reflection["q_magnitude"]
        ext = tA * OS
        x = np.linspace(-ext[0] / 2, ext[0] / 2, N, dtype=np.float32)
        y = np.linspace(-ext[1] / 2, ext[1] / 2, N, dtype=np.float32)
        z = np.linspace(-ext[2] / 2, ext[2] / 2, N, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        phase = np.zeros((N, N, N), np.float32)
        eps = cfg.STRAIN_MAGNITUDE
        if cfg.STRAIN_TYPE == "radial_gradient":
            phase += G * eps * np.sqrt(X**2 + Y**2 + Z**2)
        elif cfg.STRAIN_TYPE == "edge_dislocation":
            phase += G * (cfg.LATTICE_A / (2 * np.pi)) * np.arctan2(Y, X + 1e-10)
        elif cfg.STRAIN_TYPE == "random":
            from scipy.ndimage import gaussian_filter as gf

            np.random.seed(42)
            raw = gf(np.random.randn(N, N, N).astype(np.float32), sigma=N / 8)
            raw /= np.abs(raw).max() + 1e-12
            phase += G * eps * np.mean(tA) / 2.0 * raw
        dc = cfg.DISLOCATION
        if dc is not None:
            b = dc.b_angstrom if dc.b_angstrom else cfg.LATTICE_A
            nu = dc.nu
            px = (dc.pos_frac[0] - 0.5) * tA[0]
            py = (dc.pos_frac[1] - 0.5) * tA[1]
            if dc.line_dir == "Z":
                DX, DY = X - px, Y - py
            elif dc.line_dir == "Y":
                DX, DY = X - px, Z - py
            else:
                DX, DY = Y - px, Z - py
            R2 = DX**2 + DY**2 + 1e-20
            theta = np.arctan2(DY, DX)
            if dc.dtype == "edge":
                phase += G * (b / (2 * np.pi)) * (theta + DX * DY / (2 * (1 - nu) * R2))
            elif dc.dtype == "screw":
                phase += G * (b / (2 * np.pi)) * theta
            elif dc.dtype == "mixed":
                be = dc.b_edge if dc.b_edge else b * 0.866
                bs = dc.b_screw if dc.b_screw else b * 0.5
                phase += G * (
                    (be / (2 * np.pi)) * (theta + DX * DY / (2 * (1 - nu) * R2))
                    + (bs / (2 * np.pi)) * theta
                )
        dl = cfg.DISLOCATION_LOOP
        if dl is not None:
            b = dl.b_angstrom if dl.b_angstrom else cfg.LATTICE_A
            R_loop = dl.radius_angstrom
            lcx = (dl.center_frac[0] - 0.5) * tA[0]
            lcy = (dl.center_frac[1] - 0.5) * tA[1]
            lcz = (dl.center_frac[2] - 0.5) * tA[2]
            ndir = {
                "X": np.array([1, 0, 0]),
                "Y": np.array([0, 1, 0]),
                "Z": np.array([0, 0, 1]),
            }[dl.normal]
            d_along = (X - lcx) * ndir[0] + (Y - lcy) * ndir[1] + (Z - lcz) * ndir[2]
            perp2 = (X - lcx) ** 2 + (Y - lcy) ** 2 + (Z - lcz) ** 2 - d_along**2
            rho = np.sqrt(np.maximum(perp2, 0))
            inside = rho < R_loop
            dist2 = d_along**2 + 1e-20
            Omega = np.zeros_like(X)
            Omega[inside] = (
                2
                * np.pi
                * np.sign(d_along[inside] + 1e-30)
                * (1 - np.abs(d_along[inside]) / np.sqrt(dist2[inside] + R_loop**2))
            )
            r3 = (dist2[~inside] + rho[~inside] ** 2) ** 1.5 + 1e-30
            Omega[~inside] = np.pi * R_loop**2 * d_along[~inside] / r3
            phase += G * (b / (4 * np.pi)) * Omega
        return phase.astype(np.float32)

    def _sup(self, N, OS):
        h = 0.5 / OS
        x = np.linspace(-0.5, 0.5, N, np.float32)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        s = self.config.PARTICLE_SHAPE
        if s == "cylinder":
            return (np.sqrt(X**2 + Y**2) <= h) & (np.abs(Z) <= h)
        if s == "sphere":
            return np.sqrt(X**2 + Y**2 + Z**2) <= h
        if s == "hexagonal":
            s3 = np.sqrt(3.0) / 2.0
            return (
                np.maximum(
                    np.abs(X),
                    np.maximum(np.abs(0.5 * X + s3 * Y), np.abs(0.5 * X - s3 * Y)),
                )
                <= h
            ) & (np.abs(Z) <= h)
        return (np.abs(X) <= h) & (np.abs(Y) <= h) & (np.abs(Z) <= h)

    def simulate(self, progress_cb=None):
        c = self.config
        N = c.DETECTOR_N_PIXELS
        tA = c.particle_size_angstrom
        F2 = self.reflection["F_squared"]
        Vc = c.unit_cell_volume_ang3
        LP = self.reflection["LP_factor"]
        OS = c.TARGET_OVERSAMPLING
        dqx = (2 * np.pi / tA[0]) / OS
        dqy = (2 * np.pi / tA[1]) / OS
        dqz = (2 * np.pi / tA[2]) / OS
        qx = (np.arange(N) - N // 2).astype(np.float32) * dqx
        qy = (np.arange(N) - N // 2).astype(np.float32) * dqy
        qz = (np.arange(N) - N // 2).astype(np.float32) * dqz
        nb = c.PARTICLE_SHAPE not in ("cube", "rectangle")
        has_d = (
            c.STRAIN_TYPE != "none"
            or c.DISLOCATION is not None
            or c.DISLOCATION_LOOP is not None
        )
        if progress_cb:
            progress_cb(10)
        if has_d or nb:
            sup = self._sup(N, OS).astype(np.float32)
            ph = self._displacement(N, tA, OS)
            psi = sup * np.exp(1j * ph)
            if progress_cb:
                progress_cb(40)
            # Apply angular offset as phase ramp (shifts Bragg peak off-center)
            # AND attenuate intensity by the kinematic rocking curve envelope
            if abs(c.ANGULAR_OFFSET_DEG) > 1e-6:
                delta_th = np.radians(c.ANGULAR_OFFSET_DEG / 2.0)
                delta_q = (
                    (4 * np.pi / c.wavelength_angstrom)
                    * np.cos(self.reflection["theta_B_rad"])
                    * delta_th
                )
                # Phase ramp along first spatial axis (real-space extent = OS*tA)
                ext0 = OS * tA[0]
                rx = np.linspace(-ext0 / 2, ext0 / 2, N, dtype=np.float32)
                phase_ramp = np.exp(1j * delta_q * rx)[:, None, None]
                psi = psi * phase_ramp
            pq = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psi)))
            if progress_cb:
                progress_cb(70)
            dV = (OS * tA[0] / N) * (OS * tA[1] / N) * (OS * tA[2] / N)
            diff = (LP * F2 / Vc**2) * dV**2 * np.abs(pq) ** 2
        else:
            # For angular offset, shift the q-grid center
            qx_off = qx.copy()
            if abs(c.ANGULAR_OFFSET_DEG) > 1e-6:
                delta_th = np.radians(c.ANGULAR_OFFSET_DEG / 2.0)
                delta_q = (
                    (4 * np.pi / c.wavelength_angstrom)
                    * np.cos(self.reflection["theta_B_rad"])
                    * delta_th
                )
                qx_off = qx - delta_q
            sx = np.sinc(qx_off * tA[0] / (2 * np.pi)).astype(np.float32)
            sy = np.sinc(qy * tA[1] / (2 * np.pi)).astype(np.float32)
            sz = np.sinc(qz * tA[2] / (2 * np.pi)).astype(np.float32)
            Vp = tA[0] * tA[1] * tA[2]
            pf = LP * F2 / Vc**2 * Vp**2
            diff = (
                pf
                * (sx**2)[:, None, None]
                * (sy**2)[None, :, None]
                * (sz**2)[None, None, :]
            )
            if progress_cb:
                progress_cb(70)
        diff = self._pc(diff.astype(np.float32), dqx, dqy, dqz)
        # Angular offset: attenuate by kinematic rocking curve envelope
        # Away from exact Bragg condition, intensity drops as sinc^2
        if abs(c.ANGULAR_OFFSET_DEG) > 1e-6:
            # Shape-aware physical particle size (not just supercell box)
            # For sphere/cylinder/hexagonal, the inscribed shape is smaller than the box
            shape = c.PARTICLE_SHAPE
            if shape == "sphere":
                D_phys = float(min(tA))  # diameter = smallest box dim
            elif shape == "cylinder":
                D_phys = float(min(tA[0], tA[1]))  # diameter in basal plane
            elif shape == "hexagonal":
                D_phys = float(min(tA[0], tA[1]))  # in-plane width
            else:
                D_phys = float(np.mean(tA))  # cube/rectangle: full extent
            th_B = self.reflection["theta_B_rad"]
            # Kinematic rocking curve FWHM: dw ~ 2*lambda/(D*sin(2*theta_B))
            dw_fwhm = (
                2.0 * c.wavelength_angstrom / (D_phys * np.sin(2.0 * th_B) + 1e-12)
            )
            delta_deg = abs(c.ANGULAR_OFFSET_DEG)
            # Number of FWHM widths away
            n_fwhm = delta_deg / max(np.degrees(dw_fwhm), 1e-10)
            # Gaussian attenuation (matches kinematic rocking curve envelope)
            attenuation = np.exp(-2.0 * np.log(2) * (n_fwhm**2))
            diff *= attenuation
        if progress_cb:
            progress_cb(95)
        self.diff_volume = diff.astype(np.float32)
        self.q_grids = {"qx": qx, "qy": qy, "qz": qz}
        xh, xv, xl = self._coh()
        self.coherence = {"xi_h_A": xh, "xi_v_A": xv, "xi_l_A": xl}
        if progress_cb:
            progress_cb(100)
            return diff, self.q_grids


def add_experimental_noise(
    vol, poisson=True, readout_noise=0, air_scatter=0, dead_pixels_frac=0, seed=42
):
    rng = np.random.default_rng(seed)
    out = vol.copy().astype(np.float64)
    if air_scatter > 0:
        N = vol.shape[0]
        c = N // 2
        r0 = max(1, N // 4)
        x = np.arange(N) - c
        QX, QY, QZ = np.meshgrid(x, x, x, indexing="ij")
        r_norm = (QX**2 + QY**2 + QZ**2) / (r0**2 + 1e-10)
        peak = max(1.0, vol.max())
        out += peak * air_scatter * 1e-6 / (1.0 + r_norm**2)
    if poisson:
        scale = 1e6 / max(1, out.max())
        counts = np.clip(out * scale, 0, 1e12)
        out = rng.poisson(counts.astype(np.float64)).astype(np.float64) / scale
    if readout_noise > 0:
        out += rng.normal(0, readout_noise * out.max() * 1e-6, out.shape)
        out = np.maximum(out, 0)
    if dead_pixels_frac > 0:
        n_dead = int(dead_pixels_frac * out.size)
        idx = rng.choice(out.size, n_dead, replace=False)
        out.flat[idx] = 0
    return out.astype(np.float32)


def export_measurement(sim, refl, cfg, filepath, noise_opts=None):
    """Export simulation as .npz (like real BCDI measurement data)."""
    vol = sim.diff_volume.copy()
    if noise_opts:
        vol = add_experimental_noise(vol, **noise_opts)
    # Real-space voxel pitch in nm = (object size * oversampling) / grid_size
    N = cfg.DETECTOR_N_PIXELS
    psize_arr = np.asarray(cfg.particle_size_nm, dtype=np.float32)
    voxel_nm = psize_arr * float(cfg.TARGET_OVERSAMPLING) / float(N)
    data = {
        "diffraction_volume": vol,
        "qx": sim.q_grids["qx"],
        "qy": sim.q_grids["qy"],
        "qz": sim.q_grids["qz"],
        "hkl": np.array(refl["hkl"]),
        "d_hkl": refl["d_hkl"],
        "two_theta": np.degrees(refl["two_theta_rad"]),
        "energy_keV": cfg.BEAM_ENERGY_KEV,
        "wavelength_A": cfg.wavelength_angstrom,
        "material": cfg.MATERIAL_NAME,
        "space_group": cfg.SPACE_GROUP,
        "lattice_a": cfg.LATTICE_A,
        "lattice_b": cfg.LATTICE_B,
        "lattice_c": cfg.LATTICE_C,
        "detector_distance_m": cfg.SAMPLE_DETECTOR_DISTANCE_M,
        "pixel_size_um": cfg.DETECTOR_PIXEL_UM,
        "detector_nx": cfg.DETECTOR_NX,
        "detector_ny": cfg.DETECTOR_NY,
        "rocking_step_deg": cfg.ROCKING_STEP_DEG,
        "rocking_steps": cfg.ROCKING_STEPS,
        "beam_size_um": cfg.BEAM_SIZE_UM,
        "oversampling": cfg.TARGET_OVERSAMPLING,
        "particle_size_nm": psize_arr,
        "voxel_size_nm": voxel_nm,
    }
    if filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        import h5py

        with h5py.File(filepath, "w") as f:
            for k, v in data.items():
                if isinstance(v, str):
                    f.attrs[k] = v
                elif isinstance(v, (int, float, np.floating, np.integer)):
                    f.attrs[k] = v
                else:
                    f.create_dataset(k, data=v, compression="gzip")
    else:
        np.savez_compressed(
            filepath,
            **{
                k: (v if not isinstance(v, str) else np.array(v))
                for k, v in data.items()
            },
        )
    return filepath


# ---- Plotly helpers (unchanged from v5.9) ----
_SL = '<div id="ctrl" style="position:fixed;bottom:16px;right:16px;z-index:9999;background:rgba(13,14,18,.93);border:1px solid rgba(255,255,255,.13);border-radius:10px;padding:12px 16px;min-width:240px;font-family:Consolas,monospace;font-size:11px;color:#c0c4cc;backdrop-filter:blur(8px);box-shadow:0 4px 28px rgba(0,0,0,.55);user-select:none">'


def _sr(label, sid, mn, mx, val, step, jsfn, unit="%"):
    return f'<div style="margin-bottom:6px"><span style="display:inline-block;width:100px">{label}</span><input type="range" id="{sid}" min="{mn}" max="{mx}" value="{val}" step="{step}" style="width:95px;accent-color:#4f98a3" oninput="{jsfn}(this.value)"><span id="{sid}V" style="width:38px;display:inline-block;text-align:right">{val}{unit}</span></div>'


def _loading_html(title="Rendering 3D..."):
    """Loading overlay with progress bar + timeout fallback (never gets stuck)."""
    return f"""<div id="loadOvr" style="position:fixed;top:0;left:0;width:100%;height:100%;background:#0d0e12;z-index:99999;display:flex;align-items:center;justify-content:center;flex-direction:column"><div style="color:#4f98a3;font-size:16px;font-weight:700;margin-bottom:20px">{title}</div><div style="width:220px;height:6px;background:#161b22;border-radius:3px;overflow:hidden"><div id="loadBar" style="width:0%;height:100%;background:#4f98a3;border-radius:3px;transition:width 0.3s"></div></div><div id="loadPct" style="color:#8b949e;font-size:11px;margin-top:8px">0%</div></div><script>(function(){{var b=document.getElementById("loadBar"),t=document.getElementById("loadPct"),o=document.getElementById("loadOvr"),p=0;var iv=setInterval(function(){{p=Math.min(p+2+Math.random()*4,90);b.style.width=p+"%";t.innerText=Math.round(p)+"%"}},200);function done(){{clearInterval(iv);b.style.width="100%";t.innerText="100%";setTimeout(function(){{o.style.opacity="0";o.style.transition="opacity 0.3s";setTimeout(function(){{o.style.display="none"}},350)}},200)}};var gd=document.querySelector(".plotly-graph-div");if(gd&&gd.on)gd.on("plotly_afterplot",done);setTimeout(done,8000)}})();</script>"""


def lattice_html_with_controls(
    builder,
    max_atoms=50000,
    atom_size=3,
    show_bonds=False,
    show_strain=False,
    show_dislocation=False,
):
    import plotly.graph_objects as go

    pos, sp, disp = builder.get_lattice_atoms(max_atoms)
    if pos is None:
        return None
    CPK = {
        "Al": "#A8A8A8",
        "Cu": "#FF8C00",
        "Au": "#FFD700",
        "W": "#66CCFF",
        "Fe": "#E87060",
        "Si": "#F0C050",
        "Ge": "#99CC66",
        "Te": "#CC77CC",
        "Ga": "#C8A0A0",
        "N": "#5070F8",
        "Zn": "#8080C0",
        "O": "#FF4040",
        "Pt": "#D0D0F0",
        "C": "#808080",
    }
    sa = np.array(sp)
    traces = []
    nat = 0
    if show_dislocation and disp is not None and disp.max() > 1e-6:
        thresh = disp.max() * 0.02
        displaced_mask = disp > thresh
        for s_el in dict.fromkeys(sp):
            el_mask = sa == s_el
            undisplaced = el_mask & (~displaced_mask)
            if undisplaced.any():
                p_ = pos[undisplaced]
                traces.append(
                    go.Scatter3d(
                        x=p_[:, 0],
                        y=p_[:, 1],
                        z=p_[:, 2],
                        mode="markers",
                        name=s_el,
                        marker=dict(
                            size=max(1, atom_size - 1),
                            color=CPK.get(s_el, "#666"),
                            opacity=0.3,
                            line=dict(width=0),
                        ),
                        hoverinfo="skip",
                    )
                )
                nat += 1
        if displaced_mask.any():
            p_d = pos[displaced_mask]
            d_d = disp[displaced_mask]
            traces.append(
                go.Scatter3d(
                    x=p_d[:, 0],
                    y=p_d[:, 1],
                    z=p_d[:, 2],
                    mode="markers",
                    name="Displaced",
                    marker=dict(
                        size=atom_size + 1,
                        color=d_d,
                        colorscale="Turbo",
                        cmin=0,
                        cmax=disp.max(),
                        showscale=True,
                        colorbar=dict(title="|u| (A)", thickness=12, len=0.6, x=0.02),
                        opacity=0.95,
                        line=dict(width=0),
                    ),
                    hovertemplate="|u|=%{marker.color:.3f} A<extra></extra>",
                )
            )
            nat += 1
    elif show_strain and disp is not None and disp.max() > 1e-6:
        traces.append(
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="markers",
                name="Displacement",
                marker=dict(
                    size=atom_size,
                    color=disp,
                    colorscale="Inferno",
                    cmin=0,
                    cmax=disp.max(),
                    showscale=True,
                    colorbar=dict(title="|u| (A)", thickness=12, len=0.6, x=0.02),
                    opacity=0.9,
                    line=dict(width=0),
                ),
                hovertemplate="|u|=%{marker.color:.3f} A<extra></extra>",
            )
        )
        nat = 1
    else:
        for s in dict.fromkeys(sp):
            m = sa == s
            p_ = pos[m]
            traces.append(
                go.Scatter3d(
                    x=p_[:, 0],
                    y=p_[:, 1],
                    z=p_[:, 2],
                    mode="markers",
                    name=s,
                    marker=dict(
                        size=atom_size,
                        color=CPK.get(s, "#AAA"),
                        opacity=0.85,
                        line=dict(width=0),
                    ),
                    hovertemplate=f"<b>{s}</b><br>%{{x:.2f}},%{{y:.2f}},%{{z:.2f}}<extra></extra>",
                )
            )
            nat += 1
    bond_trace_idx = -1
    if show_bonds:
        cfg = builder.config
        M = cfg.lattice_vectors_cartesian
        nb = len(cfg.BASIS_COORDS)
        bc = np.array(
            [
                np.array(f)[0] * M[0] + np.array(f)[1] * M[1] + np.array(f)[2] * M[2]
                for f in cfg.BASIS_COORDS
            ]
        )
        if nb > 1:
            from scipy.spatial.distance import pdist

            dists = pdist(bc)
            cutoff = (
                dists[dists > 0.1].min() * 1.15
                if len(dists[dists > 0.1]) > 0
                else cfg.LATTICE_A * 0.75
            )
        else:
            cutoff = cfg.LATTICE_A * 0.75
        if len(pos) < 80000:
            from scipy.spatial import cKDTree

            pairs = cKDTree(pos).query_pairs(cutoff)
            if 0 < len(pairs) < 500000:
                bx, by, bz = [], [], []
                for i, j in pairs:
                    bx += [pos[i, 0], pos[j, 0], None]
                    by += [pos[i, 1], pos[j, 1], None]
                    bz += [pos[i, 2], pos[j, 2], None]
                traces.append(
                    go.Scatter3d(
                        x=bx,
                        y=by,
                        z=bz,
                        mode="lines",
                        name="Bonds",
                        line=dict(color="rgba(255,255,255,0.25)", width=1),
                        hoverinfo="skip",
                    )
                )
                bond_trace_idx = len(traces) - 1
    cfg = builder.config
    M = cfg.lattice_vectors_cartesian
    nx, ny, nz = cfg.SUPERCELL_MULT
    v = np.array([[i, j, k] for i in (0, nx) for j in (0, ny) for k in (0, nz)], float)
    corners = v @ M
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (4, 5),
        (4, 6),
        (3, 7),
        (5, 7),
        (6, 7),
    ]
    ex, ey, ez = [], [], []
    for i, j in edges:
        ex += [corners[i, 0], corners[j, 0], None]
        ey += [corners[i, 1], corners[j, 1], None]
        ez += [corners[i, 2], corners[j, 2], None]
    traces.append(
        go.Scatter3d(
            x=ex,
            y=ey,
            z=ez,
            mode="lines",
            line=dict(color="#FFF", width=2),
            opacity=0.5,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    sz = cfg.particle_size_nm
    ax = dict(
        backgroundcolor="#0a0d12",
        gridcolor="rgba(255,255,255,.08)",
        showbackground=True,
        color="#c0c4cc",
        zeroline=False,
    )
    dc = cfg.DISLOCATION
    dl = cfg.DISLOCATION_LOOP
    ds = ""
    if dc:
        ds = f" | disl:{dc.dtype}@({dc.pos_frac[0]:.2f},{dc.pos_frac[1]:.2f})"
    if dl:
        ds = f" | loop R={dl.radius_angstrom:.0f}A"
    fig = go.Figure(data=traces)
    # Camera: for hex/layered systems, tilt to show lateral arrangement
    cam = (
        dict(eye=dict(x=1.6, y=0.4, z=0.8))
        if cfg.LATTICE_GAMMA == 120
        else dict(eye=dict(x=1.25, y=1.25, z=1.25))
    )
    fig.update_layout(
        title=dict(
            text=f"{cfg.MATERIAL_NAME} {nx}x{ny}x{nz} {sz[0]:.1f}x{sz[1]:.1f}x{sz[2]:.1f}nm {len(pos):,}atoms{ds}",
            font=dict(size=12, color="#c0c4cc"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(**ax, title="x(A)"),
            yaxis=dict(**ax, title="y(A)"),
            zaxis=dict(**ax, title="z(A)"),
            aspectmode="data",
            camera=cam,
            bgcolor="#0a0d12",
        ),
        paper_bgcolor="#0d0e12",
        font=dict(family="Consolas,monospace", size=11, color="#c0c4cc"),
        legend=dict(
            bgcolor="rgba(15,15,20,.8)",
            bordercolor="rgba(255,255,255,.15)",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark",
    )
    html = fig.to_html(
        include_plotlyjs=True, full_html=True, config={"displaylogo": False}
    )
    # Loading overlay with fallback timeout (dismisses after 8s max)
    idxs = str(list(range(nat)))
    bond_js = ""
    bond_js_fn = ""
    if bond_trace_idx >= 0:
        bond_js = (
            '<div style="margin-bottom:6px"><label><input type="checkbox" id="bondTgl" checked onchange="tglBond(this.checked)" style="accent-color:#4f98a3"> Show bonds</label></div>'
            + _sr("Bond width", "bw", "0.5", "5", "1", "0.5", "updBw", "")
        )
        bond_js_fn = f'window.tglBond=function(on){{Plotly.restyle(gd,{{visible:on}},[{bond_trace_idx}])}};window.updBw=function(v){{document.getElementById("bwV").innerText=v;Plotly.restyle(gd,{{"line.width":Number(v)}},[{bond_trace_idx}])}};'
    strain_ctrl = ""
    if show_strain and disp is not None and disp.max() > 1e-6:
        dm = f"{disp.max():.2f}"
        strain_ctrl = _sr(
            "Strain min", "smin", "0", dm, "0", "0.01", "updSmin", "A"
        ) + _sr("Strain max", "smax", "0", dm, dm, "0.01", "updSmax", "A")
    PANEL = (
        _SL
        + '<div style="font-size:12px;font-weight:700;margin-bottom:10px;color:#4f98a3">Lattice View</div>'
        + _sr("Atom size", "asz", "1", "12", str(atom_size), "1", "updSz", "")
        + strain_ctrl
        + bond_js
        + f'</div><script>(function(){{var gd=document.querySelector(".plotly-graph-div");window.updSz=function(v){{document.getElementById("aszV").innerText=v;Plotly.restyle(gd,{{"marker.size":parseInt(v)}},{idxs})}};{bond_js_fn}if(document.getElementById("smin")){{window.updSmin=function(v){{document.getElementById("sminV").innerText=v+"A";Plotly.restyle(gd,{{"marker.cmin":Number(v)}},[0])}};window.updSmax=function(v){{document.getElementById("smaxV").innerText=v+"A";Plotly.restyle(gd,{{"marker.cmax":Number(v)}},[0])}}}}}})();</script>'
    )
    html = html.replace(
        "<body>", "<body>" + _loading_html("Rendering 3D Lattice..."), 1
    )
    return html.replace("</body>", PANEL + "</body>", 1)


def bragg_html_with_controls(sim, downsample=2, threshold_frac=1e-4, noise_opts=None):
    import plotly.graph_objects as go

    if sim.diff_volume is None:
        return None
    vol = sim.diff_volume.copy()
    if noise_opts:
        vol = add_experimental_noise(vol, **noise_opts)
    d3 = vol[::downsample, ::downsample, ::downsample]
    qx = sim.q_grids["qx"][::downsample]
    qy = sim.q_grids["qy"][::downsample]
    qz = sim.q_grids["qz"][::downsample]
    vl = np.log10(d3 + 1.0).astype(np.float32)
    vm = float(vl.max())
    QX, QY, QZ = np.meshgrid(qx, qy, qz, indexing="ij")
    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=QX.ravel(),
            y=QY.ravel(),
            z=QZ.ravel(),
            value=vl.ravel(),
            isomin=vm * 0.75,
            isomax=vm,
            opacity=1,
            surface_count=21,
            colorscale=[
                [0, "#030d1e"],
                [0.12, "#0a2a6e"],
                [0.30, "#1060c8"],
                [0.50, "#30b0e8"],
                [0.65, "#e8d830"],
                [0.80, "#e87010"],
                [0.92, "#d01808"],
                [1.0, "#ffffff"],
            ],
            showscale=True,
            colorbar=dict(
                thickness=14,
                len=0.9,
                title=dict(text="log10(I+1)", side="right"),
                tickfont=dict(size=10, color="#c0c4cc"),
            ),
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )
    cfg = sim.config
    sz = cfg.particle_size_nm
    ax = dict(
        backgroundcolor="#080a0e",
        gridcolor="rgba(255,255,255,.09)",
        showbackground=True,
        color="#c0c4cc",
        zeroline=False,
    )
    noise_str = ""
    if noise_opts:
        parts = []
        if noise_opts.get("poisson"):
            parts.append("Poisson")
        if noise_opts.get("readout_noise", 0) > 0:
            parts.append("readout")
        if noise_opts.get("air_scatter", 0) > 0:
            parts.append("air")
        if noise_opts.get("dead_pixels_frac", 0) > 0:
            parts.append("dead")
        if parts:
            noise_str = " | " + ", ".join(parts)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0e12",
        scene=dict(
            xaxis=dict(**ax, title="qx(1/A)"),
            yaxis=dict(**ax, title="qy(1/A)"),
            zaxis=dict(**ax, title="qz(1/A)"),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.7)),
            aspectmode="data",
        ),
        font=dict(family="Consolas,monospace", size=11, color="#c0c4cc"),
        title=dict(
            text=f'{cfg.MATERIAL_NAME} {sim.reflection["hkl_str"]} | {sz[0]:.0f}x{sz[1]:.0f}x{sz[2]:.0f}nm{noise_str}',
            font=dict(size=12),
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    # Fringe measurement JS
    measure_js = """<div id="meas" style="position:fixed;top:16px;right:16px;z-index:9999;background:rgba(13,14,18,.93);border:1px solid rgba(255,255,255,.13);border-radius:10px;padding:8px 12px;font-family:Consolas,monospace;font-size:11px;color:#c0c4cc;display:none">
<div style="color:#4f98a3;font-weight:700;margin-bottom:4px">Measure fringes</div>
<div id="mpts">Click 2 points on the 3D plot</div>
<div id="mdist" style="color:#3fb950;font-weight:700;margin-top:4px"></div></div>"""
    html = fig.to_html(
        include_plotlyjs=True, full_html=True, config={"displaylogo": False}
    )
    html = html.replace(
        "<body>", "<body>" + _loading_html("Rendering 3D Bragg Peak..."), 1
    )
    # Loading overlay with fallback timeout (dismisses after 8s max)
    PANEL = (
        _SL
        + '<div style="font-size:12px;font-weight:700;margin-bottom:10px;color:#4f98a3">Bragg Peak Controls</div>'
        + _sr("Iso min", "imin", "0", "100", "75", "1", "updImin")
        + _sr("Iso max", "imax", "0", "100", "100", "1", "updImax")
        + _sr("Opacity", "opac", "5", "100", "100", "5", "updOp")
        + _sr("Surfaces", "sc", "3", "50", "21", "1", "updSc", "")
        + '<div style="margin-top:8px"><label><input type="checkbox" id="measTgl" onchange="tglMeas(this.checked)" style="accent-color:#4f98a3"> Measure distance</label></div></div>'
    )
    PANEL += measure_js
    PANEL += f"""<script>(function(){{var gd=document.querySelector(".plotly-graph-div"),VM={vm:.6f};
window.updImin=function(p){{document.getElementById("iminV").innerText=p+"%";Plotly.restyle(gd,{{isomin:VM*p/100}},[0])}};
window.updImax=function(p){{document.getElementById("imaxV").innerText=p+"%";Plotly.restyle(gd,{{isomax:VM*p/100}},[0])}};
window.updOp=function(p){{var v=p/100;document.getElementById("opacV").innerText=p+"%";Plotly.restyle(gd,{{opacity:v}},[0])}};
window.updSc=function(v){{document.getElementById("scV").innerText=v;Plotly.restyle(gd,{{"surface_count":Number(v)}},[0])}};
var mp=[],measuring=false;
window.tglMeas=function(on){{measuring=on;mp=[];document.getElementById("meas").style.display=on?"block":"none";document.getElementById("mpts").innerText="Click 2 points";document.getElementById("mdist").innerText=""}};
gd.on("plotly_click",function(d){{if(!measuring||!d.points.length)return;var p=d.points[0];mp.push({{x:p.x,y:p.y,z:p.z}});
if(mp.length==1)document.getElementById("mpts").innerText="Pt1: ("+p.x.toFixed(4)+","+p.y.toFixed(4)+","+p.z.toFixed(4)+") - click 2nd";
if(mp.length>=2){{var dx=mp[1].x-mp[0].x,dy=mp[1].y-mp[0].y,dz=mp[1].z-mp[0].z,dist=Math.sqrt(dx*dx+dy*dy+dz*dz);
var d_ang=2*Math.PI/dist;document.getElementById("mdist").innerText="dq="+dist.toFixed(4)+" 1/A => d="+d_ang.toFixed(2)+" A ("+( d_ang/10).toFixed(3)+" nm)";
document.getElementById("mpts").innerText="Pt1-Pt2: dq="+dist.toFixed(4);mp=[]}}}});
}})();</script>"""
    return html.replace("</body>", PANEL + "</body>", 1)

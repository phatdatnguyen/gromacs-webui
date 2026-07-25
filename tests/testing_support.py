"""Fixtures shared by the unit tests.

Deliberately not named ``test_*`` so unittest discovery does not import it as a
test module. Everything here builds inputs from scratch, so the fast tests need
neither GROMACS nor any file from ./data.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import MDAnalysis as mda

# The application resolves ./data and ./static relative to the process working
# directory, which must therefore be the repository root, one level up from here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_GROMACS = shutil.which("gmx") is not None
requires_gromacs = unittest.skipUnless(HAS_GROMACS, "the gmx binary is not on PATH")

# Backbone atoms of one residue, with offsets in Angstrom. The arrangement is
# deliberately three-dimensional: a collinear chain makes editconf's box fitting
# produce NaN coordinates, which then breaks grompp.
_RESIDUE_ATOMS = (("N", "N", 0.00, 0.00, 0.00),
                  ("CA", "C", 1.45, 0.50, 0.30),
                  ("C", "C", 2.40, 0.00, -0.30),
                  ("O", "O", 2.60, 1.10, -0.60))


def pdb_line(serial: int, name: str, resname: str, resid: int, x: float, y: float, z: float,
             element: str, chain: str = "A") -> str:
    """Format one ATOM record with resname in columns 18-21, as NGL expects."""
    return (f"ATOM  {serial:5d} {name:<4s} {resname:<4s}{chain:1s}{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")


def write_structure_pdb(path: str, n_residues: int = 3, ions: dict[str, int] | None = None,
                        n_waters: int = 0, resname: str = "GLY") -> str:
    """Write a small protein, optional monatomic ions and optional water.

    ``ions`` maps a residue name to how many single-atom residues of it to add,
    e.g. ``{"NA": 2, "CU2P": 1}``. The residue defaults to glycine because it is
    the only amino acid with no side chain, so a backbone-only structure is
    complete as far as pdb2gmx is concerned.
    """
    ions = ions or {}
    lines: list[str] = []
    serial = 0
    resid = 0

    for residue in range(n_residues):
        resid += 1
        for name, element, dx, dy, dz in _RESIDUE_ATOMS:
            serial += 1
            lines.append(pdb_line(serial, name, resname, resid,
                                  residue * 3.8 + dx, dy, dz, element))

    for resname, count in ions.items():
        element = resname[:2] if len(resname) > 2 else resname
        for index in range(count):
            serial += 1
            resid += 1
            lines.append(pdb_line(serial, resname, resname, resid,
                                  0.0, 5.0 + index * 2.0, 0.0, element, chain="B"))

    for index in range(n_waters):
        resid += 1
        for name, element, dx in (("OW", "O", 0.0), ("HW1", "H", 0.1), ("HW2", "H", 0.2)):
            serial += 1
            lines.append(pdb_line(serial, name, "SOL", resid, dx, 0.0, 8.0 + index * 3.0, element,
                                  chain="C"))

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\nEND\n")
    return path


def write_trajectory(structure_path: str, trajectory_path: str, n_frames: int = 5,
                     step: float = 1.0) -> str:
    """Write an XTC whose every frame differs, so frame handling is observable."""
    universe = mda.Universe(structure_path)
    with mda.Writer(trajectory_path, universe.atoms.n_atoms) as writer:
        for frame in range(n_frames):
            universe.atoms.positions = universe.atoms.positions + step
            writer.write(universe.atoms)
    universe.trajectory.close()
    return trajectory_path


def frames_of(structure_path: str, trajectory_path: str) -> list:
    """Every frame's coordinates, copied because MDAnalysis reuses its buffer."""
    universe = mda.Universe(structure_path, trajectory_path)
    try:
        return [timestep.positions.copy() for timestep in universe.trajectory]
    finally:
        universe.trajectory.close()


class WorkingDirectoryTestCase(unittest.TestCase):
    """Base class giving each test a throwaway job directory under ./data.

    The callbacks resolve "./data" and "./static" relative to the process working
    directory, and path_security rejects anything outside ./data, so the tests run
    from the repository root against a real directory inside it.
    """

    def setUp(self) -> None:
        # Everything is registered with addCleanup rather than done in tearDown:
        # a subclass whose setUp fails (or calls skipTest) never reaches tearDown,
        # which would leave job directories behind in ./data.
        previous_cwd = os.getcwd()
        self.addCleanup(os.chdir, previous_cwd)

        os.chdir(REPO_ROOT)
        os.makedirs("data", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        self.addCleanup(self._clean_static)

        self._directory = tempfile.mkdtemp(prefix="_unittest_", dir="data")
        self.addCleanup(shutil.rmtree, self._directory, ignore_errors=True)
        self.working_directory_name = os.path.basename(self._directory)
        self.working_directory_path = os.path.join("data", self.working_directory_name)

    @staticmethod
    def _clean_static() -> None:
        """Remove the viewer files the callbacks write into ./static.

        Reading one of those trajectories back makes MDAnalysis cache frame offsets
        beside it as a hidden ".<name>_offsets.npz", so match with the leading dot
        stripped as well.
        """
        prefixes = ("protein_md_", "complex_md_", "_unittest_")
        for name in os.listdir(os.path.join(REPO_ROOT, "static")):
            if name.lstrip(".").startswith(prefixes):
                os.remove(os.path.join(REPO_ROOT, "static", name))

    def path(self, file_name: str) -> str:
        """Absolute-ish path of a file inside this test's job directory."""
        return os.path.join(self.working_directory_path, file_name)

    @staticmethod
    def plain_text(status: str | None) -> str:
        """Strip the HTML colour span the callbacks wrap their status in."""
        import re
        return re.sub("<[^>]+>", "", status or "")

"""Focused tests for reusable scientific and GROMACS topology helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock

import MDAnalysis as mda
import numpy as np

import utils


class PeriodicCenterOfMassTests(unittest.TestCase):
    @staticmethod
    def atoms(positions, masses):
        universe = mda.Universe.empty(len(masses), trajectory=True)
        universe.add_TopologyAttr("masses", masses)
        universe.atoms.positions = np.asarray(positions, dtype=float)
        return universe.atoms

    def test_wrapped_molecule_is_unwrapped_about_a_mass_weighted_anchor(self):
        atoms = self.atoms([[9.8, 1.0, 1.0], [0.2, 1.0, 1.0]], [12.0, 1.0])
        box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])

        centre = utils.periodic_center_of_mass(atoms, box)

        self.assertAlmostEqual(centre[0], (12.0 * 9.8 + 10.2) / 13.0, places=6)
        np.testing.assert_allclose(centre[1:], [1.0, 1.0], atol=1e-7)

    def test_whole_molecule_wider_than_half_the_box_is_not_folded_to_atom_zero(self):
        atoms = self.atoms(
            [[1.0, 1.0, 1.0], [4.0, 1.0, 1.0], [7.0, 1.0, 1.0]],
            [1.0, 1.0, 1.0],
        )
        box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])

        centre = utils.periodic_center_of_mass(atoms, box)

        np.testing.assert_allclose(centre, [4.0, 1.0, 1.0], atol=1e-7)

    def test_absent_or_invalid_box_uses_the_ordinary_center_of_mass(self):
        atoms = self.atoms([[9.8, 1.0, 1.0], [0.2, 3.0, 1.0]], [12.0, 1.0])
        expected = atoms.center_of_mass()

        for box in (None, np.zeros(6), [10.0, 10.0, 10.0]):
            with self.subTest(box=box):
                np.testing.assert_allclose(
                    utils.periodic_center_of_mass(atoms, box), expected, atol=1e-7
                )


class TopologyForceFieldTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, content: str) -> str:
        path = os.path.join(self.directory.name, "topol.top")
        with open(path, "w") as handle:
            handle.write(content)
        return path

    def write_named(self, name: str, content: str) -> str:
        path = os.path.join(self.directory.name, name)
        with open(path, "w") as handle:
            handle.write(content)
        return path

    def test_detects_name_and_family_from_forcefield_include(self):
        topology = self.write(
            '#include "../share/charmm36-jul2022.ff/forcefield.itp"\n'
            '#include "charmm36-jul2022.ff/tip3p.itp"\n'
        )
        self.assertEqual(utils.get_topology_force_field_name(topology), "charmm36-jul2022")
        self.assertEqual(utils.get_topology_force_field_family(topology), "CHARMM")

    def test_validation_accepts_family_variant_and_returns_actual_name(self):
        topology = self.write('#include "amber99sb-ildn.ff/forcefield.itp"\n')
        self.assertEqual(
            utils.validate_topology_force_field(topology, "AMBER14SB"),
            "amber99sb-ildn",
        )

    def test_validation_rejects_incompatible_cutoff_family(self):
        topology = self.write('#include "gromos54a7.ff/forcefield.itp"\n')
        with self.assertRaisesRegex(ValueError, "does not match"):
            utils.validate_topology_force_field(topology, "AMBER99SB-ILDN")

    def test_missing_or_multiple_force_fields_are_reported(self):
        topology = self.write('#include "ions.itp"\n')
        self.assertIsNone(utils.get_topology_force_field_name(topology))
        with self.assertRaisesRegex(ValueError, "has no"):
            utils.validate_topology_force_field(topology, "CHARMM36")

        topology = self.write(
            '#include "amber99sb-ildn.ff/forcefield.itp"\n'
            '#include "oplsaa.ff/forcefield.itp"\n'
        )
        with self.assertRaisesRegex(ValueError, "multiple force fields"):
            utils.get_topology_force_field_name(topology)

    def test_each_generated_family_specific_mdp_matches_its_topology(self):
        for force_field in ("AMBER99SB-ILDN", "OPLSAA", "CHARMM36", "GROMOS54A7"):
            with self.subTest(force_field=force_field):
                topology = self.write_named(
                    force_field + ".top",
                    f'#include "{force_field.lower()}.ff/forcefield.itp"\n',
                )
                mdp = self.write_named(
                    force_field + ".mdp",
                    utils.get_default_prod_md_mdp_file_content(force_field=force_field),
                )
                self.assertEqual(
                    utils.validate_mdp_topology_compatibility(mdp, topology),
                    force_field.lower(),
                )

    def test_larger_compatible_cutoffs_and_allenerpres_are_accepted(self):
        topology = self.write_named(
            "amber.top", '#include "amber14sb.ff/forcefield.itp"\n'
        )
        mdp = self.write_named(
            "custom.mdp",
            "rlist=1.3\nrvdw=1.2\nrcoulomb=1.1\nDispCorr=AllEnerPres\n"
            "coulombtype=PME\ncutoff-scheme=Verlet\n",
        )
        self.assertEqual(
            utils.validate_mdp_topology_compatibility(mdp, topology), "amber14sb"
        )

    def test_old_generic_cutoffs_are_rejected_for_gromos(self):
        topology = self.write_named(
            "gromos.top", '#include "gromos54a7.ff/forcefield.itp"\n'
        )
        mdp = self.write_named(
            "old.mdp", "rlist=1.0\nrvdw=1.0\nrcoulomb=1.0\nDispCorr=no\n"
            "coulombtype=PME\ncutoff-scheme=Verlet\n"
        )
        with self.assertRaisesRegex(ValueError, "GROMOS.*below"):
            utils.validate_mdp_topology_compatibility(mdp, topology)

    def test_amber_mdp_is_rejected_for_charmm_topology(self):
        topology = self.write_named(
            "charmm.top", '#include "charmm36.ff/forcefield.itp"\n'
        )
        mdp = self.write_named(
            "amber.mdp",
            utils.get_default_prod_md_mdp_file_content(force_field="AMBER99SB-ILDN"),
        )
        with self.assertRaisesRegex(ValueError, "CHARMM.*force-switch"):
            utils.validate_mdp_topology_compatibility(mdp, topology)

    def test_gromos_two_femtosecond_step_requires_all_bonds(self):
        topology = self.write_named(
            "gromos.top", '#include "gromos54a7.ff/forcefield.itp"\n'
        )
        common = (
            "dt=0.002\nrlist=1.0\nrvdw=1.4\nrcoulomb=1.0\n"
            "coulombtype=PME\ncutoff-scheme=Verlet\nDispCorr=no\n")
        for constraints in ("h-bonds", "none"):
            with self.subTest(constraints=constraints):
                mdp = self.write_named(
                    constraints + ".mdp", common + f"constraints={constraints}\n")
                with self.assertRaisesRegex(
                        ValueError, r"GROMOS.*constraints=.*all-bonds"):
                    utils.validate_mdp_topology_compatibility(mdp, topology)

        mdp = self.write_named(
            "all-bonds.mdp", common + "constraints=all-bonds\n")
        self.assertEqual(
            utils.validate_mdp_topology_compatibility(mdp, topology),
            "gromos54a7",
        )

    def test_gromos_one_femtosecond_step_can_keep_h_bonds(self):
        topology = self.write_named(
            "gromos.top", '#include "gromos54a7.ff/forcefield.itp"\n'
        )
        mdp = self.write_named(
            "one-fs.mdp",
            "dt=0.001\nconstraints=h-bonds\nrlist=1.0\nrvdw=1.4\n"
            "rcoulomb=1.0\ncoulombtype=Reaction-Field\n"
            "cutoff-scheme=Verlet\nDispCorr=no\n",
        )
        self.assertEqual(
            utils.validate_mdp_topology_compatibility(mdp, topology),
            "gromos54a7",
        )

    def test_gromos_minimizer_ignores_an_irrelevant_dynamics_timestep(self):
        topology = self.write_named(
            "gromos.top", '#include "gromos54a7.ff/forcefield.itp"\n'
        )
        mdp = self.write_named(
            "steep.mdp",
            "integrator=steep\ndt=0.002\nconstraints=none\nrlist=1.0\n"
            "rvdw=1.4\nrcoulomb=1.0\ncoulombtype=PME\n"
            "cutoff-scheme=Verlet\nDispCorr=no\n",
        )
        self.assertEqual(
            utils.validate_mdp_topology_compatibility(mdp, topology),
            "gromos54a7",
        )

    def test_gromos_only_applies_the_1_4_nm_floor_to_vdw(self):
        topology = self.write_named(
            "gromos.top", '#include "gromos54a7.ff/forcefield.itp"\n'
        )
        for coulombtype in ("PME", "Reaction-Field"):
            with self.subTest(coulombtype=coulombtype):
                mdp = self.write_named(
                    coulombtype + ".mdp",
                    "dt=0.002\nconstraints=all-bonds\nrlist=1.0\nrvdw=1.4\n"
                    f"rcoulomb=1.0\ncoulombtype={coulombtype}\n"
                    "cutoff-scheme=Verlet\nDispCorr=no\n",
                )
                self.assertEqual(
                    utils.validate_mdp_topology_compatibility(mdp, topology),
                    "gromos54a7",
                )

    def test_known_families_reject_cutoff_electrostatics_and_nonverlet_scheme(self):
        topology = self.write_named(
            "amber.top", '#include "amber14sb.ff/forcefield.itp"\n'
        )
        compatible = (
            "rlist=1.0\nrvdw=1.0\nrcoulomb=1.0\nDispCorr=EnerPres\n")
        cutoff = self.write_named(
            "cutoff.mdp",
            compatible + "coulombtype=Cut-off\ncutoff-scheme=Verlet\n",
        )
        with self.assertRaisesRegex(ValueError, r"AMBER.*coulombtype=Cut-off.*PME"):
            utils.validate_mdp_topology_compatibility(cutoff, topology)

        group_scheme = self.write_named(
            "group.mdp",
            compatible + "coulombtype=PME\ncutoff-scheme=Group\n",
        )
        with self.assertRaisesRegex(ValueError, r"AMBER.*cutoff-scheme=group.*Verlet"):
            utils.validate_mdp_topology_compatibility(group_scheme, topology)

        pme = self.write_named(
            "pme.mdp",
            compatible + "coulombtype=PME\ncutoff-scheme=Verlet\n",
        )
        self.assertEqual(
            utils.validate_mdp_topology_compatibility(pme, topology),
            "amber14sb",
        )

    def test_charmm_force_switch_rejects_an_extended_vdw_cutoff(self):
        topology = self.write_named(
            "charmm.top", '#include "charmm36.ff/forcefield.itp"\n'
        )
        mdp = self.write_named(
            "charmm-extended.mdp",
            "rlist=1.4\nrvdw=1.4\nrvdw-switch=1.0\nrcoulomb=1.2\n"
            "vdw-modifier=force-switch\ncoulombtype=PME\n"
            "cutoff-scheme=Verlet\nDispCorr=no\n",
        )
        with self.assertRaisesRegex(
                ValueError, r"CHARMM.*rvdw must be exactly 1.2"):
            utils.validate_mdp_topology_compatibility(mdp, topology)

    def test_custom_force_field_defers_family_policy_but_checks_numeric_syntax(self):
        topology = self.write_named(
            "custom.top", '#include "my-lab-force-field.ff/forcefield.itp"\n'
        )
        mdp = self.write_named(
            "custom.mdp",
            "dt=0.002\nrlist=0.8\nrvdw=0.8\nrcoulomb=0.8\n"
            "coulombtype=Cut-off\ncutoff-scheme=Verlet\n",
        )
        self.assertEqual(
            utils.validate_mdp_topology_compatibility(mdp, topology),
            "my-lab-force-field",
        )

        malformed = self.write_named("malformed.mdp", "dt=not-a-number\n")
        with self.assertRaisesRegex(ValueError, r"custom.*dt has invalid"):
            utils.validate_mdp_topology_compatibility(malformed, topology)


class Pdb2gmxProbeIsolationTests(unittest.TestCase):
    class FakeProcess:
        returncode = 0

        def communicate(self, input):
            return ("Select start terminus type for GLY-1\n 0: NH3+\n", "")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_each_probe_uses_distinct_output_names(self):
        commands = []

        def popen(command, **kwargs):
            commands.append(command)
            return self.FakeProcess()

        base_command = ["gmx", "pdb2gmx", "-o", "out.gro", "-p", "topol.top"]
        with unittest.mock.patch.object(utils.subprocess, "Popen", side_effect=popen):
            utils.resolve_terminus_selections(
                base_command, self.directory.name, utils.DEFAULT_TERMINUS_CHOICE, None
            )
            utils.resolve_terminus_selections(
                base_command, self.directory.name, utils.DEFAULT_TERMINUS_CHOICE, None
            )

        for flag in ("-o", "-p", "-i"):
            first = commands[0][commands[0].index(flag) + 1]
            second = commands[1][commands[1].index(flag) + 1]
            self.assertNotEqual(first, second)
            self.assertTrue(first.startswith(utils.PROBE_PDB2GMX_PREFIX + "_"))

    def test_cleanup_removes_only_the_callers_probe_files(self):
        first = utils.PROBE_PDB2GMX_PREFIX + "_first"
        second = utils.PROBE_PDB2GMX_PREFIX + "_second"
        for name in (first + ".gro", "#" + first + ".top.1#", second + ".gro"):
            with open(os.path.join(self.directory.name, name), "w") as handle:
                handle.write("temporary")

        utils._remove_pdb2gmx_probe_files(self.directory.name, first)

        self.assertFalse(os.path.exists(os.path.join(self.directory.name, first + ".gro")))
        self.assertFalse(os.path.exists(os.path.join(self.directory.name, "#" + first + ".top.1#")))
        self.assertTrue(os.path.exists(os.path.join(self.directory.name, second + ".gro")))


if __name__ == "__main__":
    unittest.main()

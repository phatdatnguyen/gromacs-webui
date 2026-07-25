"""Tests for MDP generation: cutoffs, restraints, velocities and barostat."""

from __future__ import annotations

import unittest

import utils


class CutoffSectionTests(unittest.TestCase):
    def test_charmm_uses_force_switched_vdw(self):
        section = utils.get_cutoff_mdp_section("CHARMM36")
        for setting in ("vdwtype         = cutoff",
                        "vdw-modifier    = force-switch",
                        "rvdw-switch     = 1.0",
                        "rvdw            = 1.2",
                        "rlist           = 1.2",
                        "rcoulomb        = 1.2",
                        "DispCorr        = no"):
            self.assertIn(setting, section)

    def test_non_charmm_keeps_plain_cutoffs(self):
        section = utils.get_cutoff_mdp_section("AMBER99SB-ILDN")
        self.assertIn("rvdw            = 1.0", section)
        self.assertNotIn("force-switch", section)
        self.assertNotIn("rvdw-switch", section)

    def test_charmm_detection_covers_variants_and_none(self):
        for value in ("CHARMM36", "charmm36", " charmm27 ", "CHARMM36M"):
            with self.subTest(value=value):
                self.assertTrue(utils.is_charmm_force_field(value))
        for value in ("AMBER14SB", "OPLSAA", "", None):
            with self.subTest(value=value):
                self.assertFalse(utils.is_charmm_force_field(value))

    def test_every_generator_carries_electrostatics(self):
        """No stage may silently fall back to grompp's plain cut-off default."""
        generators = (utils.get_default_ion_addition_mdp_file_content,
                      utils.get_default_energy_minimization_mdp_file_content,
                      utils.get_default_nvt_equilibration_mdp_file_content,
                      utils.get_default_npt_equilibration_mdp_file_content,
                      utils.get_default_prod_md_mdp_file_content)
        for generator in generators:
            for force_field in ("AMBER99SB-ILDN", "CHARMM36"):
                with self.subTest(generator=generator.__name__, force_field=force_field):
                    self.assertIn("coulombtype     = PME", generator(force_field=force_field))


class EquilibrationMdpTests(unittest.TestCase):
    def test_nvt_generates_velocities(self):
        content = utils.get_default_nvt_equilibration_mdp_file_content(temperature=310)
        self.assertIn("gen_vel     = yes", content)
        self.assertIn("gen_temp    = 310", content)
        self.assertIn("continuation = no", content)

    def test_nvt_and_npt_restrain_the_solute(self):
        self.assertIn("define      = -DPOSRES", utils.get_default_nvt_equilibration_mdp_file_content())
        self.assertIn("define          = -DPOSRES", utils.get_default_npt_equilibration_mdp_file_content())

    def test_npt_uses_c_rescale_but_production_uses_parrinello_rahman(self):
        self.assertIn("pcoupl          = C-rescale", utils.get_default_npt_equilibration_mdp_file_content())
        production = utils.get_default_prod_md_mdp_file_content()
        self.assertIn("pcoupl          = Parrinello-Rahman", production)
        self.assertNotIn("-DPOSRES", production)

    def test_step_count_follows_time_scale_and_step(self):
        content = utils.get_default_nvt_equilibration_mdp_file_content(time_scale_ps=100, time_step_ps=0.002)
        self.assertIn("nsteps      = 50000", content)


class ProductionMdpTests(unittest.TestCase):
    def test_initial_run_seeds_velocities_and_continuation_does_not(self):
        initial = utils.get_default_prod_md_mdp_file_content(mdp_type="Initial", random_seed=42)
        self.assertIn("gen_vel         = yes", initial)
        self.assertIn("gen_seed        = 42", initial)

        continuation = utils.get_default_prod_md_mdp_file_content(mdp_type="Continuation")
        self.assertIn("continuation    = yes", continuation)
        self.assertNotIn("gen_vel", continuation)

    def test_neural_potential_block_fixes_the_box(self):
        content = utils.get_default_prod_md_mdp_file_content(
            nnpot_active=True, nnpot_modelfile_path="models/ani2x.pt", nnpot_input_group="Protein")
        self.assertIn("nnpot-active          = true", content)
        self.assertIn("nnpot-modelfile       = models/ani2x.pt", content)
        self.assertIn("nnpot-input-group     = Protein", content)
        # The wrappers return energies but no virial, so pressure coupling is off.
        self.assertIn("pcoupl          = no", content)
        self.assertNotIn("Parrinello-Rahman", content)


if __name__ == "__main__":
    unittest.main()

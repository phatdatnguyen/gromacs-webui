"""Tests for optional neural-potential selection and GROMACS input contracts."""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import utils


class NNPotAvailabilityTests(unittest.TestCase):
    @staticmethod
    def _find_spec_with(installed: set[str]):
        return lambda name: object() if name in installed else None

    def test_dependencies_are_checked_for_the_selected_model(self):
        with mock.patch.object(
            utils.importlib.util,
            "find_spec",
            side_effect=self._find_spec_with({"torch", "torchani"}),
        ):
            self.assertEqual(utils.get_missing_nnpot_packages("ani2x"), [])
            self.assertEqual(
                utils.get_missing_nnpot_packages("mace-small"),
                ["mace", "e3nn"],
            )

    def test_general_availability_does_not_require_mace_dependencies(self):
        with mock.patch.object(
            utils.importlib.util,
            "find_spec",
            side_effect=self._find_spec_with({"torch"}),
        ), mock.patch.object(utils, "get_gromacs_nnpot_unavailable_reason", return_value=None):
            self.assertTrue(utils.is_nnpot_available())

    def test_gromacs_build_must_have_torch_support(self):
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout="GROMACS version: 2026.3\nTorch support:       disabled\n",
            stderr="",
        )
        with mock.patch.object(utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"), \
             mock.patch.object(utils.subprocess, "run", return_value=result):
            reason = utils.get_gromacs_nnpot_unavailable_reason()

        self.assertIn("Torch support: disabled", reason)

    def test_torch_enabled_gromacs_build_is_accepted(self):
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout=("GROMACS version: 2026.4\n"
                    "Torch support:       enabled (version 2.11.0)\n"),
            stderr="",
        )
        with mock.patch.object(utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"), \
             mock.patch.object(utils.subprocess, "run", return_value=result):
            self.assertIsNone(utils.get_gromacs_nnpot_unavailable_reason())

    def test_all_bundled_models_require_gromacs_2026(self):
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout=("GROMACS version: 2025.4\n"
                    "Torch support:       enabled (version 2.8.0)\n"),
            stderr="",
        )
        with mock.patch.object(
            utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"
        ), mock.patch.object(utils.subprocess, "run", return_value=result):
            for model_name in utils.SUPPORTED_NNPOT_MODELS:
                with self.subTest(model_name=model_name):
                    reason = utils.get_gromacs_nnpot_unavailable_reason(model_name)
                    self.assertIn("requires GROMACS 2026 or newer", reason)

    def test_mace_requires_2026_4_triclinic_pair_shift_fix(self):
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout=("GROMACS version: 2026.3\n"
                    "Torch support:       enabled (version 2.11.0)\n"),
            stderr="",
        )
        with mock.patch.object(
            utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"
        ), mock.patch.object(utils.subprocess, "run", return_value=result):
            self.assertIsNone(
                utils.get_gromacs_nnpot_unavailable_reason("ani2x-emle"))
            reason = utils.get_gromacs_nnpot_unavailable_reason("mace-small")

        self.assertIn("requires GROMACS 2026.4 or newer", reason)
        self.assertIn("triclinic", reason)

    def test_advanced_models_fail_closed_when_release_is_unparseable(self):
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout="Torch support:       enabled (version 2.11.0)\n",
            stderr="",
        )
        with mock.patch.object(
            utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"
        ), mock.patch.object(utils.subprocess, "run", return_value=result):
            reason = utils.get_gromacs_nnpot_unavailable_reason("mace-small")

        self.assertIn("release could not be determined", reason)

    def test_gromacs_torch_probe_is_refreshed_after_reinstall(self):
        disabled = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout="Torch support:       disabled\n",
            stderr="",
        )
        enabled = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout="Torch support:       enabled (version 2.11.0)\n",
            stderr="",
        )
        with mock.patch.object(
            utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"
        ), mock.patch.object(
            utils.subprocess, "run", side_effect=[disabled, enabled]
        ) as run:
            self.assertIn(
                "Torch support: disabled",
                utils.get_gromacs_nnpot_unavailable_reason(),
            )
            self.assertIsNone(utils.get_gromacs_nnpot_unavailable_reason())

        self.assertEqual(run.call_count, 2)

    def test_unknown_gromacs_torch_status_is_reported_accurately(self):
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout="GROMACS version: 2026.4\n",
            stderr="",
        )
        with mock.patch.object(
            utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"
        ), mock.patch.object(utils.subprocess, "run", return_value=result):
            reason = utils.get_gromacs_nnpot_unavailable_reason()

        self.assertIn("unable to determine Torch support", reason)

    def test_untrusted_model_name_is_rejected_before_it_becomes_a_path(self):
        with self.assertRaisesRegex(ValueError, "Unsupported NNPot model"):
            utils.download_nnpot_model("../../outside")

    def test_corrupt_torchscript_cache_is_quarantined_for_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "ani2x.pt")
            Path(model_path).write_bytes(b"partial archive")
            fake_torch = mock.Mock()
            fake_torch.jit.load.side_effect = RuntimeError(
                "PytorchStreamReader failed reading zip archive: failed finding central directory"
            )
            with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                usable = utils.is_cached_nnpot_model_usable("ani2x", model_path)

            self.assertFalse(usable)
            self.assertFalse(os.path.exists(model_path))
            self.assertTrue(os.path.exists(model_path + ".invalid"))

    def test_unknown_torchscript_runtime_error_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "ani2x.pt")
            Path(model_path).write_bytes(b"model")
            fake_torch = mock.Mock()
            fake_torch.jit.load.side_effect = RuntimeError("missing custom operator foo::bar")
            with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                with self.assertRaisesRegex(RuntimeError, "missing custom operator"):
                    utils.is_cached_nnpot_model_usable("ani2x", model_path)

            self.assertTrue(os.path.exists(model_path))

    def test_cache_is_rebuilt_after_exporter_package_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "ani2x.pt")
            Path(model_path).write_bytes(b"model")
            fake_torch = mock.Mock()

            def fake_load(_path, map_location, _extra_files):
                self.assertEqual(map_location, "cpu")
                _extra_files["nnpot_model_config"] = \
                    utils.get_expected_nnpot_model_config("ani2x")
                _extra_files["nnpot_package_versions"] = json.dumps(
                    {"torch": "2.7.0", "torchani": "2.8.0"},
                    sort_keys=True, separators=(",", ":"))

            fake_torch.jit.load.side_effect = fake_load
            with mock.patch.dict("sys.modules", {"torch": fake_torch}), \
                    mock.patch.object(
                        utils, "_nnpot_installed_package_versions",
                        return_value={"torch": "2.8.0", "torchani": "2.9.0"}):
                usable = utils.is_cached_nnpot_model_usable(
                    "ani2x", model_path)

            self.assertFalse(usable)
            self.assertTrue(os.path.exists(model_path + ".invalid"))


class NNPotMdpContractTests(unittest.TestCase):
    def _production_mdp(self, model_name: str) -> str:
        return utils.get_default_prod_md_mdp_file_content(
            time_step_ps=0.001,
            nnpot_active=True,
            nnpot_model_name=model_name,
            nnpot_modelfile_path=f"/models/{model_name}.pt",
        )

    def test_mace_uses_gromacs_neighbor_pairs_and_model_cutoff(self):
        content = self._production_mdp("mace-medium")

        self.assertIn("nnpot-model-input3    = nnp-charge", content)
        self.assertIn("nnpot-model-input4    = atom-pairs", content)
        self.assertIn("nnpot-model-input5    = pair-shifts", content)
        self.assertIn("nnpot-model-input6    = box", content)
        self.assertIn("nnpot-model-input7    = pbc", content)
        self.assertIn("nnpot-pair-cutoff      = 0.5", content)
        self.assertNotRegex(content, r"(?m)^pair-cutoff\s*=")

    def test_emle_uses_electrostatic_embedding_and_mm_environment(self):
        content = self._production_mdp("ani2x-emle")

        self.assertIn("nnpot-embedding       = electrostatic-model", content)
        self.assertIn("nnpot-model-input3    = atom-positions-mm", content)
        self.assertIn("nnpot-model-input4    = atom-charges-mm", content)
        self.assertIn("nnpot-model-input5    = nnp-charge", content)
        self.assertIn("nnpot-model-input6    = box", content)

    def test_ani_contract_checks_charge_then_passes_box_and_pbc(self):
        content = self._production_mdp("ani2x")

        self.assertIn("nnpot-model-input3    = nnp-charge", content)
        self.assertIn("nnpot-model-input4    = box", content)
        self.assertIn("nnpot-model-input5    = pbc", content)
        self.assertNotIn("atom-pairs", content)

    def test_unknown_model_cannot_generate_an_mdp(self):
        with self.assertRaisesRegex(ValueError, "Unsupported NNPot model"):
            self._production_mdp("not-a-model")

    def test_cache_fingerprints_invalidate_old_wrapper_contracts(self):
        self.assertIn("nonmutating-box", utils.get_expected_nnpot_model_config("ani2x"))
        self.assertIn("gromacs-pairs", utils.get_expected_nnpot_model_config("mace-small"))
        emle_config = utils.get_expected_nnpot_model_config("ani2x-emle")
        self.assertIn("neutral-gromacs-charge", emle_config)
        self.assertIn("element-check", emle_config)
        self.assertIn("runtime-device", emle_config)

    def test_generated_mdp_does_not_emit_unrecognized_scalar_charge_key(self):
        content = self._production_mdp("ani2x-emle")

        self.assertNotRegex(content, r"(?m)^nnpot-nnp-charge\s*=")
        self.assertNotRegex(content, r"(?m)^nnp-charge\s*=")

    def test_nnpot_disables_bond_constraints_and_limits_the_time_step(self):
        content = self._production_mdp("ani2x")

        self.assertRegex(content, r"(?m)^constraints\s*= none$")
        with self.assertRaisesRegex(ValueError, "no larger than.*1 fs"):
            utils.get_default_prod_md_mdp_file_content(
                time_step_ps=0.002,
                nnpot_active=True,
                nnpot_model_name="ani2x",
                nnpot_modelfile_path="/models/ani2x.pt",
            )

    def test_classical_production_keeps_force_field_constraints(self):
        content = utils.get_default_prod_md_mdp_file_content(
            time_step_ps=0.002, nnpot_active=False,
            force_field="AMBER99SB-ILDN")

        self.assertRegex(content, r"(?m)^constraints\s*= h-bonds$")


class NNPotGroupChargePreflightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.parameter_path = os.path.join(self.directory.name, "md.mdp")
        Path(self.parameter_path).write_text(
            "nnpot-active = true\n"
            "nnpot-modelfile = /models/ani2x-emle.pt\n"
            "nnpot-input-group = Protein\n",
            encoding="utf-8",
        )

    def _run(self, charges):
        probe_mdp = []

        def fake_run(command, cwd, **kwargs):
            if command[1] == "grompp":
                probe_mdp.append(Path(command[command.index("-f") + 1]).read_text(
                    encoding="utf-8"))
                Path(command[command.index("-o") + 1]).write_bytes(b"tpr")
            elif command[1] == "select":
                Path(command[command.index("-on") + 1]).write_text(
                    "[ Protein ]\n1 2\n", encoding="utf-8")
            return utils.subprocess.CompletedProcess(
                command, 0, stdout="", stderr="")

        with mock.patch.object(
            utils, "validate_working_directory",
            return_value=self.directory.name,
        ), mock.patch.object(
            utils, "run_checked_command", side_effect=fake_run,
        ), mock.patch.object(
            utils, "_stream_tpr_charges", return_value=charges,
        ):
            result = utils.validate_nnpot_input_group_charge(
                self.directory.name,
                self.parameter_path,
                os.path.join(self.directory.name, "npt.gro"),
                os.path.join(self.directory.name, "topol.top"),
                None,
                5,
            )
        return result, probe_mdp

    def test_neutral_original_group_passes_with_nnpot_disabled_probe(self):
        charge, probe_mdp = self._run({0: 0.4, 1: -0.4})

        self.assertAlmostEqual(charge, 0.0)
        self.assertEqual(len(probe_mdp), 1)
        self.assertIn("nnpot-active = false", probe_mdp[0])
        self.assertIn("nnpot-active = true", Path(
            self.parameter_path).read_text(encoding="utf-8"))

    def test_charged_original_group_is_rejected_before_nnp_grompp(self):
        with self.assertRaisesRegex(
                RuntimeError, r"total topology charge \+1\.00000 e"):
            self._run({0: 0.6, 1: 0.4})


class NNPotRuntimeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.real_runtime_version_probe = utils.get_nnpot_torch_runtime_versions
        version_probe = mock.patch.object(
            utils, "get_nnpot_torch_runtime_versions",
            return_value=("2.8.0", "2.8.1+cu128"),
        )
        version_probe.start()
        self.addCleanup(version_probe.stop)

    def test_runtime_versions_are_parsed_from_gromacs_and_python(self):
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"], 0,
            stdout=("GROMACS version: 2026.4\n"
                    "Torch support: enabled (version 2.11.0)\n"),
            stderr="",
        )
        with mock.patch.object(
            utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"
        ), mock.patch.object(
            utils.subprocess, "run", return_value=result
        ), mock.patch.object(
            utils.importlib.metadata, "version", return_value="2.8.0+cu128"
        ), mock.patch.object(
            utils, "get_nnpot_torch_runtime_versions",
            side_effect=self.real_runtime_version_probe,
        ):
            versions = utils.get_nnpot_torch_runtime_versions()

        self.assertEqual(versions, ("2.11.0", "2.8.0+cu128"))

    def test_cpu_launch_is_blocked_for_major_minor_torch_mismatch(self):
        with mock.patch.object(
            utils, "get_nnpot_torch_runtime_versions",
            return_value=("2.11.0", "2.8.0+cu128"),
        ), mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            RuntimeError, r"LibTorch 2\.11\.0.*PyTorch 2\.8\.0\+cu128"
        ):
            utils.get_nnpot_mdrun_environment(use_gpu=False)

    def test_gpu_launch_warns_but_continues_for_torch_version_skew(self):
        with mock.patch.object(
            utils, "get_nnpot_torch_runtime_versions",
            return_value=("2.11.0", "2.8.0+cu128"),
        ), mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "builtins.print"
        ) as output:
            environment = utils.get_nnpot_mdrun_environment(use_gpu=True)

        self.assertEqual(environment["GMX_NN_DEVICE"], "gpu")
        warning = " ".join(str(value) for call in output.call_args_list
                           for value in call.args)
        self.assertIn("WARNING", warning)
        self.assertIn("LibTorch 2.11.0", warning)
        self.assertIn("PyTorch 2.8.0+cu128", warning)

    def test_matching_torch_major_minor_is_allowed_on_cpu(self):
        with mock.patch.object(
            utils, "get_nnpot_torch_runtime_versions",
            return_value=("2.8.0", "2.8.1+cu128"),
        ), mock.patch.dict(os.environ, {}, clear=True):
            environment = utils.get_nnpot_mdrun_environment(use_gpu=False)

        self.assertEqual(environment["GMX_NN_DEVICE"], "cpu")

    def test_snapshot_exporter_version_overrides_current_python_version(self):
        with mock.patch.object(
            utils, "get_nnpot_torch_runtime_versions",
            return_value=("2.11.0", "2.8.0+cu128"),
        ), mock.patch.dict(os.environ, {}, clear=True):
            environment = utils.get_nnpot_mdrun_environment(
                use_gpu=False,
                exporter_torch_version="2.11.1+cu130",
            )

        self.assertEqual(environment["GMX_NN_DEVICE"], "cpu")

    def test_resolved_libnvomp_alias_is_removed_without_path_name_heuristic(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "libnvomp.so"
            runtime.touch()
            (Path(directory) / "libgomp.so.1").symlink_to(runtime.name)
            original = os.pathsep.join((directory, "/custom/gromacs/lib"))
            with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": original}):
                environment = utils.get_nnpot_mdrun_environment(use_gpu=True)

        self.assertEqual(
            environment["LD_LIBRARY_PATH"], "/custom/gromacs/lib"
        )

    def test_only_nvhpc_compiler_runtime_paths_are_removed(self):
        with tempfile.TemporaryDirectory() as compiler_runtime:
            runtime = Path(compiler_runtime) / "libnvomp.so"
            runtime.touch()
            (Path(compiler_runtime) / "libgomp.so.1").symlink_to(runtime.name)
            original = os.pathsep.join((
                "/custom/gromacs/lib",
                compiler_runtime,
                "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/lib64",
                "/custom/torch/lib",
            ))
            with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": original}):
                environment = utils.get_nnpot_mdrun_environment(use_gpu=True)
                self.assertEqual(os.environ["LD_LIBRARY_PATH"], original)

        self.assertEqual(
            environment["LD_LIBRARY_PATH"],
            os.pathsep.join((
                "/custom/gromacs/lib",
                "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/lib64",
                "/custom/torch/lib",
            )),
        )
        self.assertEqual(environment["GMX_NN_DEVICE"], "gpu")

    def test_ld_library_path_is_unset_when_every_entry_conflicts(self):
        with tempfile.TemporaryDirectory() as conflict:
            runtime = Path(conflict) / "libnvomp.so"
            runtime.touch()
            (Path(conflict) / "libgomp.so.1").symlink_to(runtime.name)
            with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": conflict}):
                environment = utils.get_nnpot_mdrun_environment(use_gpu=False)

        self.assertNotIn("LD_LIBRARY_PATH", environment)
        self.assertEqual(environment["GMX_NN_DEVICE"], "cpu")

    def test_empty_and_relative_library_path_entries_are_removed(self):
        with mock.patch.dict(
            os.environ,
            {
                "LD_LIBRARY_PATH":
                    f"/safe/lib{os.pathsep}{os.pathsep}relative",
                "LD_PRELOAD": "./job-local.so:libkeep.so",
            },
        ):
            environment = utils.get_nnpot_mdrun_environment(use_gpu=True)

        self.assertEqual(environment["LD_LIBRARY_PATH"], "/safe/lib")
        self.assertEqual(environment["LD_PRELOAD"], "libkeep.so")

    def test_an_nvhpc_looking_path_without_the_conflicting_alias_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(
                root, "nvidia", "hpc_sdk", "Linux_x86_64", "26.5",
                "compilers", "lib")
            os.makedirs(path)
            with mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": path}):
                environment = utils.get_nnpot_mdrun_environment(use_gpu=True)

        self.assertEqual(environment["LD_LIBRARY_PATH"], path)

    def test_explicit_nvidia_openmp_preloads_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            nvidia_runtime = Path(directory) / "libnvomp.so"
            nvidia_runtime.touch()
            alias = Path(directory) / "libgomp.so.1"
            alias.symlink_to(nvidia_runtime.name)
            unrelated = Path(directory) / "libkeep.so"
            unrelated.touch()
            original = f"{alias} {unrelated}:libcustom.so"
            with mock.patch.dict(
                os.environ,
                {"LD_PRELOAD": original, "LD_LIBRARY_PATH": "/custom/lib"},
            ):
                environment = utils.get_nnpot_mdrun_environment(use_gpu=True)
                self.assertEqual(os.environ["LD_PRELOAD"], original)

        self.assertEqual(
            environment["LD_PRELOAD"],
            os.pathsep.join((str(unrelated), "libcustom.so")),
        )
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/custom/lib")

    def test_named_nvidia_openmp_preload_is_removed(self):
        with mock.patch.dict(
            os.environ,
            {"LD_PRELOAD": "libnvomp.so.1:libkeep.so",
             "LD_LIBRARY_PATH": "/custom/lib"},
        ):
            environment = utils.get_nnpot_mdrun_environment(use_gpu=True)

        self.assertEqual(environment["LD_PRELOAD"], "libkeep.so")

    def test_nvhpc_host_guard_also_applies_to_preloads(self):
        with mock.patch.dict(
            os.environ,
            {"LD_PRELOAD": "libnvomp.so", "LD_LIBRARY_PATH": "/custom/lib"},
        ), mock.patch.object(
            utils, "_gromacs_host_compiler_uses_nvhpc", return_value=True
        ), self.assertRaisesRegex(RuntimeError, "NVHPC/PGI-built GROMACS"):
            utils.get_nnpot_mdrun_environment(use_gpu=True)

    def test_nvhpc_host_build_is_blocked_instead_of_losing_runtime_libraries(self):
        with tempfile.TemporaryDirectory() as conflict:
            runtime = Path(conflict) / "libnvomp.so"
            runtime.touch()
            (Path(conflict) / "libgomp.so.1").symlink_to(runtime.name)
            with mock.patch.dict(
                os.environ, {"LD_LIBRARY_PATH": conflict}
            ), mock.patch.object(
                utils, "_gromacs_host_compiler_uses_nvhpc", return_value=True
            ), self.assertRaisesRegex(RuntimeError, "NVHPC/PGI-built GROMACS"):
                utils.get_nnpot_mdrun_environment(use_gpu=True)

    def test_nvhpc_host_build_is_blocked_even_without_an_environment_hint(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            utils, "_gromacs_host_compiler_uses_nvhpc", return_value=True
        ), self.assertRaisesRegex(RuntimeError, "NVHPC/PGI-built GROMACS"):
            utils.get_nnpot_mdrun_environment(use_gpu=True)

    def test_only_host_compiler_lines_trigger_the_nvhpc_guard(self):
        cuda_only = utils.subprocess.CompletedProcess(
            ["gmx", "--version"], 0,
            stdout=("C compiler: /usr/bin/cc GNU 13\n"
                    "C++ compiler: /usr/bin/c++ GNU 13\n"
                    "CUDA compiler: /opt/nvidia/hpc_sdk/bin/nvcc (NVIDIA)\n"),
            stderr="",
        )
        nvhpc_host = utils.subprocess.CompletedProcess(
            ["gmx", "--version"], 0,
            stdout=("C compiler: /opt/nvhpc/bin/nvc NVHPC 26.5\n"
                    "C++ compiler: /opt/nvhpc/bin/nvc++ NVHPC 26.5\n"),
            stderr="",
        )
        with mock.patch.object(
            utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"
        ), mock.patch.object(
            utils.subprocess, "run", side_effect=[cuda_only, nvhpc_host]
        ):
            self.assertFalse(utils._gromacs_host_compiler_uses_nvhpc())
            self.assertTrue(utils._gromacs_host_compiler_uses_nvhpc())


class NNPotTprInspectionTests(unittest.TestCase):
    @staticmethod
    def _dump(active: str, model_file: str = "model.pt") -> str:
        if "mace-" in model_file:
            pair_cutoff = "0.5"
            inputs = (
                "atom-positions", "atom-numbers", "nnp-charge", "atom-pairs",
                "pair-shifts", "box", "pbc",
            )
        elif "ani2x-emle" in model_file:
            pair_cutoff = "0"
            inputs = (
                "atom-positions", "atom-numbers", "atom-positions-mm",
                "atom-charges-mm", "nnp-charge", "box",
            )
        else:
            pair_cutoff = "0"
            inputs = ("atom-positions", "atom-numbers", "nnp-charge", "box", "pbc")
        input_lines = "".join(
            f"       model-input{index}               = {value}\n"
            for index, value in enumerate(inputs, start=1)
        )
        return (
            "inputrec:\n"
            "   dt                         = 0.001\n"
            "   pcoupl                     = No\n"
            "     nnpot:\n"
            f"       active                     = {active}\n"
            f"       modelfile                  = {model_file}\n"
            "       input-group                = Protein\n"
            f"       pair-cutoff                = {pair_cutoff}\n"
            f"       embedding                  = {'electrostatic-model' if 'ani2x-emle' in model_file else 'mechanical'}\n"
            "       nnp-charge                 = 0\n"
            f"{input_lines}"
            "     pull:\n"
            "       ngroup                     = 0\n"
        )

    def _inspect(self, output: str):
        process = utils.subprocess.CompletedProcess(
            ["gmx", "dump"], 0, stdout=output, stderr="")
        with mock.patch.object(
            utils, "validate_working_directory", return_value="/safe/job"
        ), mock.patch.object(
            utils, "validate_local_file_path", return_value="/safe/job/md.tpr"
        ), mock.patch.object(
            utils.os.path, "isfile", return_value=True
        ), mock.patch.object(
            utils, "run_checked_command", return_value=process
        ) as run, mock.patch.object(
            utils, "verify_nnpot_tpr_charge_attestation", return_value=0.0
        ), mock.patch.object(
            utils, "require_nnpot_model_snapshot_for_bundled_model",
            return_value=None,
        ):
            result = utils.inspect_tpr_nnpot_configuration("/safe/job", "md.tpr")
        self.assertEqual(
            run.call_args.args[0], ["gmx", "dump", "-s", "/safe/job/md.tpr"])
        return result

    def _require_classical(self, output: str) -> None:
        process = utils.subprocess.CompletedProcess(
            ["gmx", "dump"], 0, stdout=output, stderr="")
        with mock.patch.object(
            utils, "validate_working_directory", return_value="/safe/job"
        ), mock.patch.object(
            utils, "validate_local_file_path", return_value="/safe/job/md.tpr"
        ), mock.patch.object(
            utils.os.path, "isfile", return_value=True
        ), mock.patch.object(
            utils, "run_checked_command", return_value=process
        ) as run:
            utils.require_classical_tpr(
                "/safe/job", "md.tpr", "the protein-only workflow")
        self.assertEqual(
            run.call_args.args[0], ["gmx", "dump", "-s", "/safe/job/md.tpr"])

    def test_active_mace_tpr_is_detected_from_the_tpr_not_the_ui(self):
        self.assertEqual(
            self._inspect(self._dump("true", "/models/mace-small.pt")),
            (True, "/models/mace-small.pt"),
        )

    def test_ordinary_tpr_is_detected_as_non_nnpot(self):
        self.assertEqual(
            self._inspect(self._dump("false")),
            (False, "model.pt"),
        )

    def test_dump_without_nnpot_section_is_an_ordinary_tpr(self):
        self.assertEqual(
            utils._parse_tpr_nnpot_dump("inputrec from an older GROMACS"),
            (False, None),
        )

    def test_protein_only_guard_accepts_a_classical_tpr(self):
        self._require_classical(self._dump("false"))

    def test_protein_only_guard_rejects_an_nnpot_tpr(self):
        with self.assertRaisesRegex(
                RuntimeError, "Protein-Ligand Complex workflow"):
            self._require_classical(self._dump("true", "/models/ani2x.pt"))

    def test_malformed_nnpot_section_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Could not determine"):
            utils._parse_tpr_nnpot_dump("     nnpot:\n       modelfile = model.pt")

    def test_tpr_state_overrides_a_stale_ui_checkbox(self):
        with mock.patch.object(
            utils, "inspect_tpr_nnpot_configuration",
            return_value=(True, "/models/mace-small.pt"),
        ), mock.patch.object(
            utils, "get_gromacs_nnpot_unavailable_reason", return_value=None,
        ), mock.patch.object(
            utils, "validate_tpr_nnpot_elements",
        ):
            self.assertTrue(
                utils.resolve_nnpot_launch_state("/safe/job", "md.tpr", False)
            )

    def test_custom_model_without_provenance_does_not_claim_current_torch(self):
        metadata = {}
        with mock.patch.object(
            utils, "inspect_tpr_nnpot_configuration",
            return_value=(True, "/custom/research-model.pt"),
        ), mock.patch.object(
            utils, "get_gromacs_nnpot_unavailable_reason", return_value=None,
        ), mock.patch.object(
            utils, "validate_tpr_nnpot_elements"
        ), mock.patch.object(
            utils, "get_nnpot_model_exporter_torch_version",
            return_value=None,
        ):
            self.assertTrue(utils.resolve_nnpot_launch_state(
                "/safe/job", "md.tpr", True, launch_metadata=metadata))

        self.assertEqual(metadata, {})

    def test_launch_rechecks_the_current_gromacs_model_compatibility(self):
        with mock.patch.object(
            utils, "inspect_tpr_nnpot_configuration",
            return_value=(True, "/models/mace-small.pt"),
        ), mock.patch.object(
            utils, "get_gromacs_nnpot_unavailable_reason",
            return_value="mace-small requires GROMACS 2026.4 or newer",
        ) as compatibility, mock.patch.object(
            utils, "validate_tpr_nnpot_elements",
        ) as validate_group, self.assertRaisesRegex(
            RuntimeError, "requires GROMACS 2026.4"
        ):
            utils.resolve_nnpot_launch_state("/safe/job", "md.tpr", True)

        compatibility.assert_called_once_with("mace-small")
        validate_group.assert_not_called()

    def test_pressure_coupling_is_rejected_for_active_nnpot(self):
        dump = self._dump("true", "/models/ani2x.pt").replace(
            "pcoupl                     = No",
            "pcoupl                     = Parrinello-Rahman")

        with self.assertRaisesRegex(RuntimeError, "pcoupl = no"):
            utils._validate_tpr_nnpot_contract(dump)

    def test_timestep_larger_than_one_femtosecond_is_rejected(self):
        dump = self._dump("true", "/models/ani2x.pt").replace(
            "dt                         = 0.001",
            "dt                         = 0.002")

        with self.assertRaisesRegex(RuntimeError, r"no larger than.*1 fs"):
            utils._validate_tpr_nnpot_contract(dump)

    def test_missing_or_nonpositive_timestep_is_rejected(self):
        missing = self._dump("true", "/models/ani2x.pt").replace(
            "   dt                         = 0.001\n", "")
        zero = self._dump("true", "/models/ani2x.pt").replace(
            "dt                         = 0.001",
            "dt                         = 0")

        for dump in (missing, zero):
            with self.subTest(dump=dump), self.assertRaisesRegex(
                    RuntimeError, "positive timestep"):
                utils._validate_tpr_nnpot_contract(dump)

    def test_old_mace_contract_without_pairs_or_cutoff_is_rejected(self):
        dump = self._dump("true", "/models/mace-small.pt")
        dump = dump.replace("pair-cutoff                = 0.5", "pair-cutoff                = 0")
        dump = dump.replace("       model-input4               = atom-pairs\n", "")
        dump = dump.replace("       model-input5               = pair-shifts\n", "")

        with self.assertRaisesRegex(RuntimeError, "stale model inputs"):
            utils._validate_tpr_nnpot_contract(dump)

    def test_emle_requires_electrostatic_embedding(self):
        dump = self._dump("true", "/models/ani2x-emle.pt").replace(
            "embedding                  = electrostatic-model",
            "embedding                  = mechanical")

        with self.assertRaisesRegex(RuntimeError, "electrostatic-model"):
            utils._validate_tpr_nnpot_contract(dump)

    def test_non_emle_models_require_mechanical_embedding(self):
        dump = self._dump("true", "/models/ani2x.pt").replace(
            "embedding                  = mechanical",
            "embedding                  = electrostatic-model")

        with self.assertRaisesRegex(RuntimeError, "requires.*mechanical"):
            utils._validate_tpr_nnpot_contract(dump)

    def test_content_addressed_model_filename_is_recognized(self):
        dump = self._dump("true", "/models/ani2x-" + "a" * 64 + ".pt")

        self.assertEqual(utils._validate_tpr_nnpot_contract(dump), "ani2x")

    def test_custom_model_with_bundled_prefix_is_not_misclassified(self):
        self.assertIsNone(
            utils.get_nnpot_model_name_from_path("/models/ani2x-solvent.pt"))

    def test_neutral_models_reject_a_charged_nnp_region(self):
        dump = self._dump("true", "/models/ani2x.pt").replace(
            "nnp-charge                 = 0",
            "nnp-charge                 = -1")

        with self.assertRaisesRegex(RuntimeError, "require neutral NNP regions"):
            utils._validate_tpr_nnpot_contract(dump)

    def test_emle_rejects_a_nonzero_nnp_region_charge(self):
        dump = self._dump("true", "/models/ani2x-emle.pt").replace(
            "nnp-charge                 = 0",
            "nnp-charge                 = 1")

        with self.assertRaisesRegex(RuntimeError, "require neutral NNP regions"):
            utils._validate_tpr_nnpot_contract(dump)


class NNPotElementPreflightTests(unittest.TestCase):
    DUMP = (
        "inputrec:\n"
        "   pcoupl = No\n"
        "     nnpot:\n"
        "       active = true\n"
        "       modelfile = /models/ani1x.pt\n"
        "       input-group = Protein_LIG\n"
        "     pull:\n"
    )

    def _validate(self, atomic_numbers, model_file="/models/ani1x.pt",
                  touching_constraints=0):
        dump_process = utils.subprocess.CompletedProcess(
            ["gmx", "dump"], 0, stdout=self.DUMP, stderr="")

        def fake_run(command, cwd, **kwargs):
            if command[1] == "dump":
                return dump_process
            if command[1] == "select":
                index_path = command[command.index("-on") + 1]
                Path(index_path).write_text(
                    "[ Protein_LIG ]\n1 2\n", encoding="utf-8")
                return utils.subprocess.CompletedProcess(
                    command, 0, stdout="", stderr="")
            self.fail(f"Unexpected GROMACS command: {command}")

        with mock.patch.object(
            utils, "validate_working_directory", return_value="/safe/job"
        ), mock.patch.object(
            utils, "validate_local_file_path", return_value="/safe/job/md.tpr"
        ), mock.patch.object(
            utils, "run_checked_command", side_effect=fake_run
        ), mock.patch.object(
            utils, "_stream_tpr_nnpot_group_data",
            return_value=(atomic_numbers, touching_constraints)
        ):
            utils.validate_tpr_nnpot_elements(
                "/safe/job", "md.tpr", model_file)

    def test_missing_atomic_number_is_rejected_before_mdrun(self):
        with self.assertRaisesRegex(RuntimeError, "without valid atomic numbers"):
            self._validate({0: 6, 1: -1})

    def test_model_specific_unsupported_element_is_named(self):
        with self.assertRaisesRegex(RuntimeError, r"unsupported by ani1x: S \(Z=16\)"):
            self._validate({0: 6, 1: 16})

    def test_supported_elements_pass(self):
        self.assertIsNone(self._validate({0: 6, 1: 8}))

    def test_constraint_touching_nnp_group_is_rejected(self):
        with self.assertRaisesRegex(
                RuntimeError, r"3 constraint interaction\(s\) touching"):
            self._validate({0: 6, 1: 8}, touching_constraints=3)

    def test_custom_model_still_gets_constraint_validation(self):
        with self.assertRaisesRegex(RuntimeError, "constraints = none"):
            self._validate(
                {0: 6, 1: 8}, model_file="/custom/research-model.pt",
                touching_constraints=3)

    def test_expanded_tpr_constraint_parser_catches_boundary_atoms(self):
        self.assertEqual(
            utils._constraint_atom_indices_from_tpr_dump_line(
                "  7 type=279 (CONSTR)  12  98\n"),
            (12, 98),
        )
        self.assertEqual(
            utils._constraint_atom_indices_from_tpr_dump_line(
                "  4 type=282 (SETTLE)  76  77  78\n"),
            (76, 77, 78),
        )
        self.assertIsNone(utils._constraint_atom_indices_from_tpr_dump_line(
            "functype[279]=CONSTR, dA=0.1, dB=0.1\n"))

    def test_emle_uses_its_narrower_hcnos_species_table(self):
        self.assertEqual(
            utils.NNPOT_MODEL_ATOMIC_NUMBERS["ani2x-emle"],
            frozenset({1, 6, 7, 8, 16}),
        )
        with self.assertRaisesRegex(
                RuntimeError, r"unsupported by ani2x-emle: F \(Z=9\)"):
            self._validate(
                {0: 6, 1: 9}, model_file="/models/ani2x-emle.pt")


class NNPotWrapperSignatureTests(unittest.TestCase):
    """Keep MDP input order exactly synchronized with TorchScript forward args.

    The optional ML stack is intentionally absent from the normal test
    environment, so parse the wrapper source without importing torch.
    """

    @classmethod
    def setUpClass(cls):
        source_path = Path(__file__).resolve().parents[1] / "nnpot_models.py"
        cls.tree = ast.parse(source_path.read_text(encoding="utf-8"))

    def _forward_arguments(self, class_name: str) -> list[str]:
        class_node = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        forward = next(
            node for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        return [argument.arg for argument in forward.args.args[1:]]

    @staticmethod
    def _mdp_inputs(model_name: str) -> list[str]:
        section = utils.get_nnpot_model_input_mdp_section(model_name)
        numbered_inputs = []
        for line in section.splitlines():
            match = re.match(r"nnpot-model-input(\d+)\s*=\s*(\S+)", line)
            if match:
                numbered_inputs.append((int(match.group(1)), match.group(2)))
        return [value for _, value in sorted(numbered_inputs)]

    def test_every_wrapper_signature_matches_its_mdp_input_order(self):
        contracts = {
            "ani1x": (
                "GmxANI1xModel",
                ["positions", "atomic_numbers", "nnp_charge", "box", "pbc"],
                ["atom-positions", "atom-numbers", "nnp-charge", "box", "pbc"],
            ),
            "ani2x": (
                "GmxANI2xModel",
                ["positions", "atomic_numbers", "nnp_charge", "box", "pbc"],
                ["atom-positions", "atom-numbers", "nnp-charge", "box", "pbc"],
            ),
            "mace-medium": (
                "GmxMACEModel",
                ["positions", "atomic_numbers", "nnp_charge", "pairs", "shifts", "cell", "pbc"],
                ["atom-positions", "atom-numbers", "nnp-charge", "atom-pairs", "pair-shifts", "box", "pbc"],
            ),
            "ani2x-emle": (
                "GmxANI2xEMLEModel",
                ["positions_nn", "atomic_numbers", "positions_mm", "charges_mm", "nnp_charge", "cell"],
                ["atom-positions", "atom-numbers", "atom-positions-mm", "atom-charges-mm", "nnp-charge", "box"],
            ),
        }

        for model_name, (class_name, wrapper_args, mdp_inputs) in contracts.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(self._forward_arguments(class_name), wrapper_args)
                self.assertEqual(self._mdp_inputs(model_name), mdp_inputs)

    def test_emle_passes_a_validated_integer_charge_to_its_scripted_model(self):
        class_node = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "GmxANI2xEMLEModel"
        )
        forward = next(
            node for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        source = ast.unparse(forward)

        self.assertIn("requires an integer NNP-region charge", source)
        self.assertIn("qm_charge = int(rounded_charge.item())", source)
        self.assertIn("self.model._device = device", source)
        self.assertIn("self.model._emle._device = device", source)
        self.assertIn("self.model._emle._emle_base._device = device", source)


if __name__ == "__main__":
    unittest.main()

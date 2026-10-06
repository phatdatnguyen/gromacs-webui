"""Contracts for background-process ownership and reliable cancellation."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BackgroundProcessLaunchTests(unittest.TestCase):
    def test_every_managed_background_process_owns_a_private_session(self):
        targets = {
            "protein_md_simulation.py": {
                "on_run_nvt_equilibration",
                "on_run_npt_equilibration",
                "on_run_prod_md",
                "on_continue_prod_md",
            },
            "protein_ligand_complex_md_simulation.py": {
                "on_run_nvt_equilibration",
                "on_run_npt_equilibration",
                "on_run_prod_md",
                "on_continue_prod_md",
                "on_run_mmpbsa",
            },
        }

        for file_name, function_names in targets.items():
            tree = ast.parse((PROJECT_ROOT / file_name).read_text(encoding="utf-8"))
            functions = {
                node.name: node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for function_name in function_names:
                with self.subTest(file=file_name, function=function_name):
                    function = functions[function_name]
                    launches = [
                        node for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "Popen"
                    ]
                    self.assertEqual(len(launches), 1)
                    keyword = next(
                        (item for item in launches[0].keywords
                         if item.arg == "start_new_session"),
                        None,
                    )
                    self.assertIsNotNone(keyword)
                    self.assertIsInstance(keyword.value, ast.Constant)
                    self.assertIs(keyword.value.value, True)

    def test_nnpot_production_children_receive_sanitized_environment(self):
        targets = {"protein_ligand_complex_md_simulation.py"}
        for file_name in targets:
            tree = ast.parse((PROJECT_ROOT / file_name).read_text(encoding="utf-8"))
            functions = {
                node.name: node for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }
            for function_name in ("on_run_prod_md", "on_continue_prod_md"):
                with self.subTest(file=file_name, function=function_name):
                    function = functions[function_name]
                    helper_calls = [
                        node for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "get_nnpot_mdrun_environment"
                    ]
                    self.assertEqual(len(helper_calls), 1)
                    tpr_state_calls = [
                        node for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "resolve_nnpot_launch_state"
                    ]
                    self.assertEqual(len(tpr_state_calls), 1)
                    launch = next(
                        node for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "Popen"
                    )
                    environment = next(
                        (item for item in launch.keywords if item.arg == "env"),
                        None,
                    )
                    self.assertIsNotNone(environment)

    def test_nnpot_grompp_receives_the_same_sanitized_environment(self):
        for file_name in ("protein_ligand_complex_md_simulation.py",):
            tree = ast.parse(
                (PROJECT_ROOT / file_name).read_text(encoding="utf-8"))
            function = next(
                node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_on_generate_prod_md_tpr_file_reserved")
            environment_builds = [
                node for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_nnpot_mdrun_environment"
            ]
            grompp_calls = [
                node for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_grompp_with_gromos_warning_policy"
            ]

            with self.subTest(file=file_name):
                self.assertEqual(len(environment_builds), 1)
                self.assertEqual(len(grompp_calls), 1)
                environment_keyword = next(
                    (item for item in grompp_calls[0].keywords
                     if item.arg == "environment"),
                    None,
                )
                self.assertIsNotNone(environment_keyword)
                self.assertIsInstance(environment_keyword.value, ast.Name)
                self.assertEqual(
                    environment_keyword.value.id, "nnpot_environment")

    def test_protein_mdp_generator_cannot_enable_nnpot(self):
        """The protein-only public MDP callback has no positive NNPot path."""
        source = (PROJECT_ROOT / "protein_md_simulation.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "on_generate_prod_md_mdp_file")

        argument_names = {argument.arg for argument in function.args.args}
        self.assertFalse(
            argument_names & {
                "nnpot_active", "nnpot_model_name", "nnpot_input_group",
            })
        mdp_builds = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_default_prod_md_mdp_file_content"
        ]
        self.assertEqual(len(mdp_builds), 1)
        keyword_names = {keyword.arg for keyword in mdp_builds[0].keywords}
        self.assertFalse(
            keyword_names & {
                "nnpot_active", "nnpot_model_name",
                "nnpot_modelfile_path", "nnpot_input_group",
            })
        self.assertNotIn("Use Machine Learning Potential (NNPot)", source)

    def test_protein_grompp_rejects_nnpot_mdp_before_launch(self):
        tree = ast.parse((PROJECT_ROOT / "protein_md_simulation.py").read_text(
            encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_on_generate_prod_md_tpr_file_reserved")
        guards = [
            node for node in ast.walk(function)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "mdp_uses_nnpot"
                for child in ast.walk(node.test)
            )
            and any(isinstance(child, ast.Raise) for child in ast.walk(node))
        ]
        self.assertEqual(len(guards), 1)
        grompp_launch = next(
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_grompp_with_gromos_warning_policy")
        self.assertLess(guards[0].lineno, grompp_launch.lineno)

    def test_protein_mdrun_rejects_nnpot_tpr_before_launch(self):
        tree = ast.parse((PROJECT_ROOT / "protein_md_simulation.py").read_text(
            encoding="utf-8"))
        functions = {
            node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for function_name in ("on_run_prod_md", "on_continue_prod_md"):
            with self.subTest(function=function_name):
                function = functions[function_name]
                guards = [
                    node for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "require_classical_tpr"
                ]
                self.assertEqual(len(guards), 1)
                launch = next(
                    node for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "Popen"
                )
                self.assertLess(guards[0].lineno, launch.lineno)


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

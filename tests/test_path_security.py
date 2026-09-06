import asyncio
import inspect
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import path_security


class PathSecurityTests(unittest.TestCase):
    def test_file_name_rejects_path_components(self):
        for value in ("../secret", "subdir/file.pdb", "/tmp/file.pdb", r"..\secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                path_security.validate_file_name(value)

    def test_file_name_accepts_plain_name(self):
        self.assertEqual(path_security.validate_file_name("protein.pdb"), "protein.pdb")

    def test_file_extension_contract_is_case_insensitive_and_role_specific(self):
        path_security.validate_file_extension(
            "protein.PDB", (".pdb",), "protein file")
        with self.assertRaisesRegex(ValueError, r"expected \.mdp"):
            path_security.validate_file_extension(
                "valuable.tpr", (".mdp",), "parameter file")

    def test_file_name_rejects_empty_controls_and_topology_quotes(self):
        for value in ("", "   ", " file.pdb", "file.pdb ", "bad\nname.pdb",
                      "bad\x00name.pdb", 'bad"name.itp', "bad'name.itp",
                      "bad<name.pdb", "bad>name.pdb"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                path_security.validate_file_name(value)

    def test_local_file_requires_a_name(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as directory:
            with self.assertRaises(ValueError):
                path_security.validate_local_file_path(directory, None)

    def test_working_directory_rejects_outside_data_root(self):
        with self.assertRaises(ValueError):
            path_security.validate_working_directory("/tmp")

    def test_working_directory_rejects_symlink_escape(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as directory:
            link = Path(directory) / "outside"
            link.symlink_to("/tmp", target_is_directory=True)
            with self.assertRaises(ValueError):
                path_security.validate_working_directory(link)

    def test_local_file_rejects_symlink_escape(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as directory:
            link = Path(directory) / "output.pdb"
            link.symlink_to("/tmp/outside.pdb")
            with self.assertRaises(ValueError):
                path_security.validate_local_file_path(directory, "output.pdb")

    def test_callback_validates_client_controlled_state_and_filename(self):
        called = False

        def callback(working_directory_path, output_file_name):
            nonlocal called
            called = True

        secured = path_security.secure_working_directory_callback(callback)
        with self.assertRaises(ValueError):
            secured(str(path_security.DATA_ROOT / "job"), "../../outside")
        self.assertFalse(called)

    def test_static_asset_names_are_safe_and_unique_per_render(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as directory:
            first = path_security.static_asset_basename("protein_md_structure", directory)
            second = path_security.static_asset_basename("protein_md_structure", directory)

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("protein_md_structure_"))
        self.assertNotIn(Path(directory).name, first)
        with self.assertRaises(ValueError):
            path_security.static_asset_basename("../escape", path_security.DATA_ROOT)

    def test_static_cleanup_only_removes_old_generated_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            static_root = Path(temporary_directory)
            old_generated = static_root / "protein_md_trajectory_deadbeef_view.html"
            fresh_generated = static_root / "complex_md_structure_deadbeef.pdb"
            unrelated = static_root / "site.html"
            wrong_extension = static_root / "protein_md_structure_deadbeef.css"
            for path in (old_generated, fresh_generated, unrelated, wrong_extension):
                path.write_text("x")
            old_timestamp = time.time() - 7200
            os.utime(old_generated, (old_timestamp, old_timestamp))

            with mock.patch.object(path_security, "STATIC_ROOT", static_root):
                removed = path_security.cleanup_stale_static_assets(3600)

            self.assertEqual(removed, 1)
            self.assertFalse(old_generated.exists())
            self.assertTrue(fresh_generated.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(wrong_extension.exists())

    def test_static_cleanup_rejects_invalid_age(self):
        for value in (-1, float("inf"), float("nan"), "not-a-number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                path_security.cleanup_stale_static_assets(value)

    def test_static_cleanup_enforces_live_byte_and_file_quotas(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            static_root = Path(temporary_directory)
            generated = [
                static_root / f"protein_md_structure_deadbeef_{index}.html"
                for index in range(4)
            ]
            for index, path in enumerate(generated):
                path.write_bytes(b"x" * 10)
                timestamp = time.time() - (100 - index)
                os.utime(path, (timestamp, timestamp))
            unrelated = static_root / "site.html"
            unrelated.write_bytes(b"x" * 100)

            with mock.patch.object(path_security, "STATIC_ROOT", static_root):
                removed = path_security.cleanup_stale_static_assets(
                    1000, max_total_bytes=20, max_files=2)

            self.assertEqual(removed, 2)
            self.assertFalse(generated[0].exists())
            self.assertFalse(generated[1].exists())
            self.assertTrue(generated[2].exists() and generated[3].exists())
            self.assertTrue(unrelated.exists())

    def test_static_cleanup_is_opportunistic_when_directory_cannot_be_listed(self):
        with mock.patch.object(path_security.Path, "iterdir", side_effect=PermissionError):
            self.assertEqual(path_security.cleanup_stale_static_assets(), 0)

    def test_callback_tags_dataframe_with_its_source_job(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as directory:
            def callback(working_directory_path):
                return pd.DataFrame({"value": [1]})

            frame = path_security.secure_working_directory_callback(callback)(directory)

        self.assertEqual(
            frame.attrs[path_security.DATAFRAME_WORKING_DIRECTORY_ATTR],
            str(Path(directory).resolve()),
        )

    def test_generator_results_are_tagged_with_their_source_job(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as directory:
            def callback(working_directory_path):
                yield ("status", pd.DataFrame({"value": [1]}))

            result = next(path_security.secure_working_directory_callback(callback)(directory))

        self.assertEqual(
            result[1].attrs[path_security.DATAFRAME_WORKING_DIRECTORY_ATTR],
            str(Path(directory).resolve()),
        )

    def test_callback_tagging_does_not_mutate_callback_owned_outputs(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as directory:
            frame = pd.DataFrame({"value": [1]})
            update = {"__type__": "update", "value": [frame], "visible": True}

            def callback(working_directory_path):
                return update

            result = path_security.secure_working_directory_callback(callback)(directory)

        self.assertIsNot(result, update)
        self.assertIsNot(result["value"], update["value"])
        self.assertIsNot(result["value"][0], frame)
        self.assertNotIn(path_security.DATAFRAME_WORKING_DIRECTORY_ATTR, frame.attrs)
        self.assertEqual(result["__type__"], "update")
        self.assertTrue(result["visible"])
        self.assertEqual(
            result["value"][0].attrs[path_security.DATAFRAME_WORKING_DIRECTORY_ATTR],
            str(Path(directory).resolve()),
        )

    def test_async_callback_kinds_and_results_are_preserved(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as directory:
            async def coroutine_callback(working_directory_path):
                return pd.DataFrame({"value": [1]})

            async def generator_callback(working_directory_path):
                yield pd.DataFrame({"value": [2]})

            secured_coroutine = path_security.secure_working_directory_callback(
                coroutine_callback)
            secured_generator = path_security.secure_working_directory_callback(
                generator_callback)
            self.assertTrue(inspect.iscoroutinefunction(secured_coroutine))
            self.assertTrue(inspect.isasyncgenfunction(secured_generator))

            coroutine_result = asyncio.run(secured_coroutine(directory))

            async def first_result():
                return await anext(secured_generator(directory))

            generator_result = asyncio.run(first_result())

        expected = str(Path(directory).resolve())
        self.assertEqual(
            coroutine_result.attrs[path_security.DATAFRAME_WORKING_DIRECTORY_ATTR], expected)
        self.assertEqual(
            generator_result.attrs[path_security.DATAFRAME_WORKING_DIRECTORY_ATTR], expected)

    def test_callback_rejects_dataframe_from_another_job(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with (tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as source,
              tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as destination):
            frame = pd.DataFrame({"value": [1]})
            frame = path_security.tag_dataframe_provenance(frame, source)

            called = False

            def callback(working_directory_path, df):
                nonlocal called
                called = True

            secured = path_security.secure_working_directory_callback(callback)
            with self.assertRaisesRegex(ValueError, "different working directory"):
                secured(destination, frame)
            self.assertFalse(called)

    def test_callback_rejects_dataframe_nested_in_another_job_output(self):
        path_security.DATA_ROOT.mkdir(exist_ok=True)
        with (tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as source,
              tempfile.TemporaryDirectory(dir=path_security.DATA_ROOT) as destination):
            frame = path_security.tag_dataframe_provenance(
                pd.DataFrame({"value": [1]}), source)

            def callback(working_directory_path, result_state):
                return result_state

            secured = path_security.secure_working_directory_callback(callback)
            with self.assertRaisesRegex(ValueError, "different working directory"):
                secured(destination, {"value": [frame]})


if __name__ == "__main__":
    unittest.main()

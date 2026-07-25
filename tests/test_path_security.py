import tempfile
import unittest
from pathlib import Path

import path_security


class PathSecurityTests(unittest.TestCase):
    def test_file_name_rejects_path_components(self):
        for value in ("../secret", "subdir/file.pdb", "/tmp/file.pdb", r"..\secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                path_security.validate_file_name(value)

    def test_file_name_accepts_plain_name(self):
        self.assertEqual(path_security.validate_file_name("protein.pdb"), "protein.pdb")

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


if __name__ == "__main__":
    unittest.main()

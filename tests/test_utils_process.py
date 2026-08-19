"""Tests for command execution, run shutdown, GPU flags and shared run state."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

import utils

GROMACS_STYLE_FAILURE = textwrap.dedent(
    """
    import sys
    sys.stderr.write('''                  :-) GROMACS - gmx grompp, 2026.2 (-:
    Executable:   /usr/local/gromacs/bin/gmx
    -------------------------------------------------------
    Program:     gmx grompp, version 2026.2
    Source file: src/gromacs/gmxpreprocess/grompp.cpp (line 123)

    Fatal error:
    Atom OW in residue SOL not found in rtp entry
    For more information and tips for troubleshooting, please check the GROMACS
    website at https://manual.gromacs.org/current/user-guide/run-time-errors.html
    -------------------------------------------------------
    ''')
    sys.exit(1)
    """
)


class RunCheckedCommandTests(unittest.TestCase):
    def python(self, source: str) -> list[str]:
        return [sys.executable, "-c", textwrap.dedent(source)]

    def test_returns_completed_process_on_success(self):
        result = utils.run_checked_command(self.python("print('hello')"))
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)

    def test_failure_message_keeps_the_diagnostic_and_drops_the_banner(self):
        with self.assertRaises(Exception) as caught:
            utils.run_checked_command(self.python(GROMACS_STYLE_FAILURE))
        message = str(caught.exception)

        self.assertIn("Fatal error", message)
        self.assertIn("Atom OW in residue SOL not found", message)
        # the version banner and the boilerplate footer are noise
        self.assertNotIn("Executable:", message)
        self.assertNotIn("manual.gromacs.org", message)
        self.assertIn("exit status 1", message)

    def test_failure_without_a_known_marker_falls_back_to_the_tail(self):
        with self.assertRaises(Exception) as caught:
            utils.run_checked_command(self.python("""
                import sys
                for i in range(60):
                    sys.stderr.write(f'line {i}\\n')
                sys.exit(2)
                """))
        message = str(caught.exception)
        self.assertIn("line 59", message)      # the end of the output survives
        self.assertNotIn("line 0", message)    # the beginning is trimmed
        self.assertIn("exit status 2", message)

    def test_stdin_input_is_forwarded(self):
        result = utils.run_checked_command(
            self.python("import sys; print('got', sys.stdin.read().strip())"), stdin_input="7\n")
        self.assertIn("got 7", result.stdout)

    def test_cwd_is_honoured(self):
        with tempfile.TemporaryDirectory() as directory:
            result = utils.run_checked_command(self.python("import os; print(os.getcwd())"), cwd=directory)
            self.assertEqual(os.path.realpath(result.stdout.strip()), os.path.realpath(directory))

    def test_missing_executable_still_raises(self):
        with self.assertRaises(Exception):
            utils.run_checked_command(["definitely-not-a-real-binary-xyz"])


class StopProcessGracefullyTests(unittest.TestCase):
    def spawn(self, source: str) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", textwrap.dedent(source)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self._reap, proc)
        return proc

    @staticmethod
    def _reap(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    def test_cooperative_process_is_terminated_not_killed(self):
        proc = self.spawn("import time; time.sleep(30)")
        utils.stop_process_gracefully(proc)
        self.assertIsNotNone(proc.poll())
        # negative return code == died from a signal; SIGTERM, not SIGKILL
        self.assertEqual(proc.returncode, -15)

    @unittest.skipUnless(hasattr(__import__("signal"), "SIGTERM"), "POSIX signals required")
    def test_process_ignoring_sigterm_is_eventually_killed(self):
        proc = self.spawn("""
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print('ready', flush=True)
            time.sleep(30)
            """)
        proc.stdout.readline()                     # wait until the handler is installed
        started = time.time()
        utils.stop_process_gracefully(proc, timeout=1)
        self.assertIsNotNone(proc.poll())
        self.assertEqual(proc.returncode, -9)
        self.assertLess(time.time() - started, 10)

    def test_already_finished_and_none_are_no_ops(self):
        proc = self.spawn("pass")
        proc.wait()
        utils.stop_process_gracefully(proc)        # must not raise
        utils.stop_process_gracefully(None)


class GpuOptionTests(unittest.TestCase):
    def test_no_flags_when_gpu_is_off(self):
        self.assertEqual(utils.get_gpu_mdrun_options(False, 1), [])

    def test_single_rank_offloads_nonbonded_and_pme(self):
        self.assertEqual(utils.get_gpu_mdrun_options(True, 1), ["-nb", "gpu", "-pme", "gpu"])

    def test_multiple_ranks_keep_pme_on_the_cpu(self):
        """GPU PME is not implemented for more than one PME rank."""
        self.assertEqual(utils.get_gpu_mdrun_options(True, 4), ["-nb", "gpu"])

    def test_never_offloads_tasks_that_clash_with_position_restraints(self):
        for ranks in (1, 2, 8):
            options = utils.get_gpu_mdrun_options(True, ranks)
            self.assertNotIn("-bonded", options)
            self.assertNotIn("-update", options)

    def test_cpu_only_options_pin_every_task_to_the_cpu(self):
        """mdrun offloads to a detected GPU unless each task is asked for by name."""
        options = utils.get_cpu_only_mdrun_options()
        for task in ("-nb", "-pme", "-bonded"):
            self.assertEqual(options[options.index(task) + 1], "cpu")
        self.assertNotIn("gpu", options)


class ProcessStateTests(unittest.TestCase):
    def test_starts_idle_with_a_lock(self):
        state = utils.ProcessStateDict()
        self.assertIsNone(state["proc"])
        self.assertFalse(state["running"])
        with state["lock"]:
            pass

    def test_deepcopy_gets_a_fresh_lock_and_clean_state(self):
        """gr.State deep-copies its value, and a copied lock would not protect anything."""
        state = utils.ProcessStateDict()
        state["running"] = True
        copied = copy.deepcopy(state)
        self.assertFalse(copied["running"])
        self.assertIsNot(copied["lock"], state["lock"])


if __name__ == "__main__":
    unittest.main()

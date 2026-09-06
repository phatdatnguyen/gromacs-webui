"""Tests for command execution, run shutdown, GPU flags and shared run state."""

from __future__ import annotations

import copy
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import unittest.mock

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

    def test_too_many_warnings_failure_includes_the_hidden_warning(self):
        stderr = """WARNING 1 [file topol.top, line 7]:
  Unsupported atom naming was detected.

Routine progress that separates the warning from the final error.
Fatal error:
Too many warnings (1).
"""
        completed = subprocess.CompletedProcess(
            ["gmx", "grompp"], 1, stdout="", stderr=stderr)
        with unittest.mock.patch.object(
                utils, "run_managed_command", return_value=completed), \
                self.assertRaises(Exception) as caught:
            utils.run_checked_command(["gmx", "grompp"])

        message = str(caught.exception)
        self.assertIn("Unsupported atom naming", message)
        self.assertIn("Too many warnings (1)", message)

    def test_numbered_grompp_input_error_is_not_hidden_by_fatal_footer(self):
        stderr = """ERROR 1 [file ions.top, line 20]:
  No such moleculetype K

Fatal error:
There was 1 error in input file(s)
"""
        completed = subprocess.CompletedProcess(
            ["gmx", "grompp"], 1,
            stdout="Routine progress that must not displace the diagnostic\n",
            stderr=stderr)
        with unittest.mock.patch.object(
                utils, "run_managed_command", return_value=completed), \
                self.assertRaises(Exception) as caught:
            utils.run_checked_command(["gmx", "grompp"])

        message = str(caught.exception)
        self.assertIn("No such moleculetype K", message)
        self.assertIn("There was 1 error", message)
        self.assertNotIn("Routine progress", message)

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

    def test_registry_race_stops_child_spawned_before_registration_failed(self):
        with unittest.mock.patch.object(
                utils, "register_process_job",
                side_effect=RuntimeError("reservation disappeared")), \
                unittest.mock.patch.object(utils, "stop_process_gracefully") as stop:
            with self.assertRaisesRegex(RuntimeError, "reservation disappeared"):
                utils.run_checked_command(self.python("import time; time.sleep(30)"))
        self.assertEqual(stop.call_count, 1)
        self.assertEqual(stop.call_args.kwargs, {"timeout": 0})
        self.assertFalse(utils.is_working_directory_busy(os.getcwd()))
        # The mocked stop deliberately leaves the process alive; reap it here.
        proc = stop.call_args.args[0]
        proc.kill()
        proc.wait()

    def test_large_command_output_is_spooled_and_returned_with_a_memory_cap(self):
        with unittest.mock.patch.object(
                utils.tempfile, "SpooledTemporaryFile",
                side_effect=AssertionError("output must not spill to disk")):
            result = utils.run_checked_command(self.python("""
                import sys
                sys.stdout.write('BEGIN-' + 'x' * (5 * 1024 * 1024) + '-END')
                """))
        self.assertTrue(result.stdout.startswith("BEGIN-"))
        self.assertTrue(result.stdout.endswith("-END"))
        self.assertIn("output truncated", result.stdout)
        self.assertLessEqual(
            len(result.stdout.encode("utf-8")),
            utils.MAX_CAPTURED_COMMAND_OUTPUT_BYTES)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_registered_checked_command_is_stopped_during_server_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            ready_path = os.path.join(directory, "ready")
            result = []

            command = self.python(f"""
                import pathlib
                import signal
                import subprocess
                import sys
                import time
                child = subprocess.Popen([
                    sys.executable, '-c',
                    'import signal,time; '
                    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                    'time.sleep(30)'])
                pathlib.Path({ready_path!r}).write_text(str(child.pid))
                time.sleep(30)
            """)

            def invoke():
                try:
                    utils.run_checked_command(command, cwd=directory)
                except Exception as exc:
                    result.append(exc)

            worker = __import__("threading").Thread(target=invoke)
            worker.start()
            deadline = time.monotonic() + 5
            while not os.path.exists(ready_path) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(os.path.exists(ready_path), "command did not start")
            with open(ready_path) as handle:
                child_pid = int(handle.read())

            try:
                self.assertEqual(
                    utils.stop_all_registered_processes(timeout=0.2), 1)
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())
                self.assertTrue(result)
                try:
                    with open(f"/proc/{child_pid}/stat") as handle:
                        state = handle.read().split()[2]
                except FileNotFoundError:
                    state = None
                self.assertIn(state, (None, "Z", "X"))
            finally:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                worker.join(timeout=3)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_natural_leader_exit_does_not_leave_an_orphaned_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            child_path = os.path.join(directory, "child.pid")
            command = self.python(f"""
                import pathlib
                import signal
                import subprocess
                import sys
                child = subprocess.Popen(
                    [sys.executable, '-c',
                     'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                pathlib.Path({child_path!r}).write_text(str(child.pid))
            """)

            result = utils.run_managed_command(command, cwd=directory)
            self.assertEqual(result.returncode, 0)
            with open(child_path) as handle:
                child_pid = int(handle.read())
            try:
                with open(f"/proc/{child_pid}/stat") as handle:
                    state = handle.read().split()[2]
            except FileNotFoundError:
                state = None
            self.assertIn(state, (None, "Z", "X"))


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
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

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

    def test_exit_race_during_terminate_is_a_successful_no_op(self):
        class JustExitedProcess:
            def poll(self):
                return None

            def terminate(self):
                raise ProcessLookupError("already exited")

        utils.stop_process_gracefully(JustExitedProcess())

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_private_process_group_receives_term_then_kill(self):
        class GroupProcess:
            pid = 424242

            def __init__(self):
                self.wait_calls = 0

            def poll(self):
                return None

            def terminate(self):
                raise AssertionError("private process group should receive the signal")

            def kill(self):
                raise AssertionError("private process group should receive the signal")

            def wait(self, timeout=None):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired("gmx", timeout)
                return -9

        proc = GroupProcess()
        with unittest.mock.patch.object(utils.os, "getpgid", return_value=proc.pid), \
                unittest.mock.patch.object(utils.os, "killpg") as killpg:
            utils.stop_process_gracefully(proc, timeout=0.01)

        self.assertEqual(
            killpg.call_args_list,
            [unittest.mock.call(proc.pid, utils.signal.SIGTERM),
             unittest.mock.call(proc.pid, utils.signal.SIGKILL)],
        )

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_private_session_stop_also_terminates_and_reaps_a_child(self):
        source = """
            import signal
            import subprocess
            import sys
            import time

            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

            def stop_with_child(*_):
                child.wait(timeout=2)
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, stop_with_child)
            print(child.pid, flush=True)
            time.sleep(30)
        """
        proc = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        child_pid = int(proc.stdout.readline())
        try:
            utils.stop_process_gracefully(proc, timeout=5)
            self.assertEqual(proc.returncode, 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._reap(proc)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_private_session_with_cooperative_child_returns_promptly(self):
        source = """
            import subprocess
            import sys
            import time

            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            print(child.pid, flush=True)
            time.sleep(30)
        """
        proc = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        child_pid = int(proc.stdout.readline())
        started = time.monotonic()
        try:
            utils.stop_process_gracefully(proc, timeout=2)
            self.assertLess(time.monotonic() - started, 1)
            self.assertEqual(proc.returncode, -signal.SIGTERM)
            try:
                with open(f"/proc/{child_pid}/stat") as handle:
                    state = handle.read().split()[2]
            except FileNotFoundError:
                state = None
            self.assertIn(state, (None, "Z", "X"))
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._reap(proc)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_private_session_kills_a_descendant_that_ignores_sigterm(self):
        source = """
            import signal
            import subprocess
            import sys
            import time

            child = subprocess.Popen([
                sys.executable, "-c",
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(30)",
            ], stdout=subprocess.PIPE, text=True)
            child.stdout.readline()
            print(child.pid, flush=True)
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            time.sleep(30)
        """
        proc = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        child_pid = int(proc.stdout.readline())
        try:
            utils.stop_process_gracefully(proc, timeout=0.2)
            self.assertEqual(proc.returncode, 0)

            # SIGKILL may briefly leave a grandchild zombie for PID 1 to reap,
            # but it must no longer be a runnable writer when the call returns.
            try:
                with open(f"/proc/{child_pid}/stat") as handle:
                    state = handle.read().split()[2]
            except FileNotFoundError:
                state = None
            self.assertIn(state, (None, "Z", "X"))
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._reap(proc)

    def test_process_without_a_pid_uses_portable_popen_methods(self):
        class PortableProcess:
            def __init__(self):
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return -15

        proc = PortableProcess()
        utils.stop_process_gracefully(proc)
        self.assertTrue(proc.terminated)

    def test_non_integer_mock_pid_is_never_treated_as_an_owned_process_group(self):
        proc = unittest.mock.MagicMock()
        proc.pid = unittest.mock.MagicMock()
        proc._gromacs_webui_process_group = unittest.mock.MagicMock()
        with unittest.mock.patch.object(utils.os, "getpgid") as getpgid:
            self.assertIsNone(utils._owned_process_group(proc))
        getpgid.assert_not_called()


class GpuOptionTests(unittest.TestCase):
    def test_gpu_off_names_the_cpu_rather_than_staying_silent(self):
        """Passing nothing leaves every task on "auto", which picks a found GPU."""
        for ranks in (1, 2, 8):
            options = utils.get_mdrun_hardware_options(False, ranks)
            self.assertNotIn("gpu", options)
            for task in ("-nb", "-pme", "-bonded"):
                self.assertEqual(options[options.index(task) + 1], "cpu")

    def test_single_rank_offloads_nonbonded_and_pme(self):
        self.assertEqual(utils.get_mdrun_hardware_options(True, 1), ["-nb", "gpu", "-pme", "gpu"])

    def test_multiple_ranks_keep_pme_on_the_cpu(self):
        """GPU PME is not implemented for more than one PME rank."""
        self.assertEqual(utils.get_mdrun_hardware_options(True, 4), ["-nb", "gpu"])

    def test_never_offloads_tasks_that_clash_with_position_restraints(self):
        for ranks in (1, 2, 8):
            options = utils.get_mdrun_hardware_options(True, ranks)
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

    class FakeProcess:
        def __init__(self, returncode=None):
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def test_watcher_only_clears_the_exact_process_it_watched(self):
        """A's late watcher must not clear B after a stop/start (the ABA race)."""
        state = utils.ProcessStateDict()
        old_proc = self.FakeProcess(0)
        new_proc = self.FakeProcess(None)
        old_key = utils.get_process_job_key("/tmp/job-a", "md")
        new_key = utils.get_process_job_key("/tmp/job-b", "md")

        utils.set_process_running(state, old_proc, old_key, "old run", "/tmp/job-a")
        utils.set_process_running(state, new_proc, new_key, "new run", "/tmp/job-b")
        utils.watch_process(old_proc, state, old_key)

        self.assertTrue(state["running"])
        self.assertIs(state["proc"], new_proc)
        self.assertIsNone(state["completion_status"])
        self.assertFalse(utils.clear_process_state_if_current(state, old_proc))
        self.assertIs(state["proc"], new_proc)

    def test_natural_completion_records_the_exit_code_and_named_status(self):
        state = utils.ProcessStateDict()
        proc = self.FakeProcess(7)
        key = utils.get_process_job_key("/tmp/job", "nvt")
        utils.set_process_running(state, proc, key, "NVT equilibration", "/tmp/job",
                                  "See nvt.log for details.")

        utils.watch_process(proc, state, key)

        self.assertFalse(state["running"])
        self.assertEqual(state["returncode"], 7)
        self.assertIn("NVT equilibration", state["completion_status"])
        self.assertIn("exit code 7", state["completion_status"])
        self.assertIn("nvt.log", state["completion_status"])
        running, message, color, directory = utils.consume_process_completion(state)
        self.assertFalse(running)
        self.assertEqual(color, "red")
        self.assertEqual(directory, os.path.realpath("/tmp/job"))
        self.assertIn("exit code 7", message)
        self.assertFalse(state["completion_pending"])

    def test_an_intentional_stop_is_not_reported_as_a_crash_to_other_sessions(self):
        state = utils.ProcessStateDict()
        proc = self.FakeProcess(-15)
        proc._gromacs_webui_stopped_by_user = True
        key = utils.get_process_job_key("/tmp/job", "md")
        utils.set_process_running(state, proc, key, "Production MD", "/tmp/job")

        utils.watch_process(proc, state, key)

        self.assertIn("stopped by user", state["completion_status"])
        self.assertNotIn("failed with exit code", state["completion_status"])

    def test_registry_rejects_a_second_writer_and_release_is_identity_checked(self):
        key = utils.get_process_job_key("/tmp", f"registry-{id(self)}")
        first = self.FakeProcess(None)
        replacement = self.FakeProcess(None)
        claimed, active = utils.reserve_process_job(key)
        self.assertTrue(claimed)
        self.assertIsNone(active)
        utils.register_process_job(key, first)
        self.addCleanup(utils.release_process_job, key, first)

        claimed, active = utils.reserve_process_job(key)
        self.assertFalse(claimed)
        self.assertIs(active, first)

        # A stale watcher carrying another process object cannot free the slot.
        utils.release_process_job(key, replacement)
        claimed, active = utils.reserve_process_job(key)
        self.assertFalse(claimed)
        self.assertIs(active, first)

    def test_switching_directories_detaches_without_stopping_the_old_process(self):
        state = utils.ProcessStateDict()
        proc = self.FakeProcess(None)
        key = utils.get_process_job_key("/tmp/old-job", "md")
        utils.set_process_running(state, proc, key, "Production MD", "/tmp/old-job")

        stopped_proc, stopped_key = utils.clear_process_state_for_directory(
            state, "/tmp/new-job")

        self.assertIsNone(stopped_proc)
        self.assertIsNone(stopped_key)
        self.assertFalse(state["running"])
        self.assertIsNone(proc.poll(), "detaching the UI must leave the child alive")

    def test_directory_busy_check_is_path_aware_and_ignores_finished_jobs(self):
        active_key = utils.get_process_job_key("/tmp/job", "md")
        finished_key = utils.get_process_job_key("/tmp/finished", "md")
        for key, proc in ((active_key, self.FakeProcess(None)),
                          (finished_key, self.FakeProcess(0))):
            claimed, _ = utils.reserve_process_job(key)
            self.assertTrue(claimed)
            utils.register_process_job(key, proc)
            self.addCleanup(utils.release_process_job, key, proc)

        self.assertTrue(utils.is_working_directory_busy("/tmp/job"))
        self.assertFalse(utils.is_working_directory_busy("/tmp/job-with-similar-prefix"))
        self.assertFalse(utils.is_working_directory_busy("/tmp/finished"))

    def test_shutdown_clears_reservations_and_stops_each_distinct_process_once(self):
        class ManagedProcess:
            def __init__(self):
                self.returncode = None
                self.terminate_calls = 0
                self.wait_saw_unlocked_registry = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminate_calls += 1

            def wait(self, timeout=None):
                acquired = utils._PROCESS_REGISTRY_LOCK.acquire(blocking=False)
                self.wait_saw_unlocked_registry = acquired
                if acquired:
                    utils._PROCESS_REGISTRY_LOCK.release()
                self.returncode = -15
                return self.returncode

        proc = ManagedProcess()
        keys = [utils.get_process_job_key("/tmp/shutdown-job", suffix)
                for suffix in ("md", "duplicate-reference")]
        reservation = utils.get_process_job_key("/tmp/shutdown-reservation", "launching")
        claimed, _ = utils.reserve_process_job(keys[0])
        self.assertTrue(claimed)
        utils.register_process_job(keys[0], proc)
        # Simulate two registry references to the same process; the public API
        # intentionally serializes writers within one directory now.
        with utils._PROCESS_REGISTRY_LOCK:
            utils._PROCESS_REGISTRY[keys[1]] = proc
        claimed, _ = utils.reserve_process_job(reservation)
        self.assertTrue(claimed)
        self.assertTrue(utils.is_working_directory_busy("/tmp/shutdown-job"))

        stopped = utils.stop_all_registered_processes(timeout=0.01)

        self.assertEqual(stopped, 1)
        self.assertEqual(proc.terminate_calls, 1)
        self.assertTrue(proc.wait_saw_unlocked_registry)
        self.assertFalse(utils.is_working_directory_busy("/tmp/shutdown-job"))
        # The launch reservation was atomically cleared as well.
        claimed, active = utils.reserve_process_job(reservation)
        self.assertTrue(claimed)
        self.assertIsNone(active)
        utils.release_process_job(reservation)

    def test_shutdown_timeout_is_one_global_budget_not_one_per_process(self):
        class StubbornProcess:
            def __init__(self):
                self.killed = False

            def poll(self):
                return -9 if self.killed else None

            def terminate(self):
                pass

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                if self.killed:
                    return -9
                if timeout:
                    time.sleep(timeout)
                raise subprocess.TimeoutExpired("job", timeout)

        processes = [StubbornProcess() for _ in range(3)]
        for index, proc in enumerate(processes):
            key = utils.get_process_job_key(
                f"/tmp/global-shutdown-budget-{index}", "job")
            claimed, _ = utils.reserve_process_job(key)
            self.assertTrue(claimed)
            utils.register_process_job(key, proc)

        started = time.monotonic()
        stopped = utils.stop_all_registered_processes(timeout=0.12)
        elapsed = time.monotonic() - started

        self.assertEqual(stopped, 3)
        self.assertTrue(all(proc.killed for proc in processes))
        self.assertLess(elapsed, 0.25)


if __name__ == "__main__":
    unittest.main()

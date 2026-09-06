"""Aggregate host-resource admission for simulations in different job folders."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as protein_workflow
import utils
from tests.testing_support import WorkingDirectoryTestCase


class _LiveProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminate_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = -15
        return self.returncode


class _CompletedProcess(_LiveProcess):
    def __init__(self, returncode: int = 0) -> None:
        super().__init__()
        self.returncode = returncode

    def wait(self, timeout: float | None = None) -> int:
        assert self.returncode is not None
        return self.returncode


class _ExitedGroupLeader(_CompletedProcess):
    def __init__(self, process_group: int = 765432) -> None:
        super().__init__(0)
        self.pid = process_group
        self._gromacs_webui_process_group = process_group


class ResourceLedgerTests(unittest.TestCase):
    """The registry lock must make global CPU/GPU admission indivisible."""

    def setUp(self) -> None:
        self._temporary_root = tempfile.TemporaryDirectory(
            prefix="gromacs-webui-resource-")
        self.addCleanup(self._temporary_root.cleanup)
        self._keys: list[str] = []

    def key(self, name: str) -> str:
        directory = os.path.join(self._temporary_root.name, name)
        os.makedirs(directory, exist_ok=True)
        key = utils.get_process_job_key(directory, "run")
        self._keys.append(key)
        self.addCleanup(utils.release_process_job, key)
        return key

    def claim(self, key: str) -> None:
        claimed, active = utils.reserve_process_job(key)
        self.assertTrue(claimed)
        self.assertIsNone(active)

    def register(self, key: str, proc: _LiveProcess) -> None:
        utils.register_process_job(key, proc)
        self.addCleanup(utils.release_process_job, key, proc)

    def run_contenders(self, requests: list[tuple[str, int, bool]]) -> list[
            tuple[str, str | Exception]]:
        barrier = threading.Barrier(len(requests) + 1)
        outcomes: list[tuple[str, str | Exception]] = []
        outcomes_lock = threading.Lock()

        def contend(key: str, cpu_slots: int, use_gpu: bool) -> None:
            try:
                claimed, active = utils.reserve_process_job(key)
                if not claimed or active is not None:
                    raise AssertionError("independent job directory was not reserved")
                barrier.wait(timeout=5)
                result: str | Exception = utils.reserve_process_resources(
                    key, cpu_slots, 1, use_gpu)
                outcome = "admitted"
            except Exception as exc:  # Captured for assertions in the main thread.
                result = exc
                outcome = "rejected"
            with outcomes_lock:
                outcomes.append((outcome, result))

        threads = [
            threading.Thread(target=contend, args=request)
            for request in requests
        ]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        return outcomes

    def test_cpu_capacity_uses_tightest_physical_affinity_and_cgroup_cap(self):
        with mock.patch.object(
                utils.psutil, "cpu_count",
                side_effect=lambda logical=True: 16 if logical else 8), \
                mock.patch.object(utils.os, "sched_getaffinity",
                                  return_value=set(range(6))), \
                mock.patch.object(utils, "_get_cgroup_cpu_capacity",
                                  return_value=4):
            self.assertEqual(utils.get_mdrun_cpu_capacity(), 4)

    def test_cgroup_v2_uses_the_tightest_ancestor_quota_and_rounds_down(self):
        system_files = {
            "/proc/self/cgroup": "0::/tenant/job\n",
            "/proc/self/mountinfo": (
                "29 23 0:26 / /sys/fs/cgroup rw,nosuid - "
                "cgroup2 cgroup rw\n"),
            "/sys/fs/cgroup/tenant/job/cpu.max": "450000 100000\n",
            "/sys/fs/cgroup/tenant/cpu.max": "250000 100000\n",
            "/sys/fs/cgroup/cpu.max": "max 100000\n",
        }
        with mock.patch.object(
                utils, "_read_optional_system_text",
                side_effect=lambda path: system_files.get(path)):
            self.assertEqual(utils._get_cgroup_cpu_capacity(), 2)

    def test_concurrent_jobs_cannot_overbook_the_aggregate_cpu_budget(self):
        first, second = self.key("cpu-a"), self.key("cpu-b")
        with mock.patch.object(utils, "get_mdrun_cpu_capacity", return_value=4):
            outcomes = self.run_contenders([
                (first, 3, False),
                (second, 3, False),
            ])

        admitted = [value for outcome, value in outcomes if outcome == "admitted"]
        rejected = [value for outcome, value in outcomes if outcome == "rejected"]
        self.assertEqual(len(admitted), 1, outcomes)
        self.assertEqual(len(rejected), 1, outcomes)
        self.assertIsInstance(rejected[0], utils.ResourceAdmissionError)
        message = str(rejected[0])
        self.assertIn("requests 3 CPU slots", message)
        self.assertIn("4-slot budget", message)
        self.assertIn("only 1 remain", message)
        self.assertIn("Stop or wait", message)

    def test_concurrent_jobs_cannot_both_reserve_the_gpu(self):
        first, second = self.key("gpu-a"), self.key("gpu-b")
        with mock.patch.object(utils, "get_mdrun_cpu_capacity", return_value=4):
            outcomes = self.run_contenders([
                (first, 1, True),
                (second, 1, True),
            ])

        admitted = [value for outcome, value in outcomes if outcome == "admitted"]
        rejected = [value for outcome, value in outcomes if outcome == "rejected"]
        self.assertEqual(len(admitted), 1, outcomes)
        self.assertEqual(len(rejected), 1, outcomes)
        self.assertIsInstance(rejected[0], utils.ResourceAdmissionError)
        self.assertIn("GPU is already reserved", str(rejected[0]))
        self.assertIn("clear Use GPU", str(rejected[0]))

    def test_attaching_is_idempotent_and_does_not_charge_resources_twice(self):
        running_key = self.key("running")
        other_key = self.key("other")
        proc = _LiveProcess()
        with mock.patch.object(utils, "get_mdrun_cpu_capacity", return_value=4):
            self.claim(running_key)
            utils.reserve_process_resources(running_key, 3, 1, True)
            self.register(running_key, proc)

            claimed, active = utils.reserve_process_job(running_key)
            self.assertFalse(claimed)
            self.assertIs(active, proc)
            # A refreshed session sees the original admission; it cannot add a
            # second three-slot/GPU charge for the same registered key.
            self.assertEqual(
                utils.reserve_process_resources(running_key, 1, 1, False),
                "Reserved 3 CPU slots and exclusive GPU use.")

            self.claim(other_key)
            with self.assertRaisesRegex(utils.ResourceAdmissionError,
                                        "only 1 remain"):
                utils.reserve_process_resources(other_key, 2, 1, False)

    def test_identity_checked_release_frees_resources_for_reuse(self):
        running_key = self.key("release-a")
        next_key = self.key("release-b")
        proc = _LiveProcess()
        replacement = _LiveProcess()
        with mock.patch.object(utils, "get_mdrun_cpu_capacity", return_value=3):
            self.claim(running_key)
            utils.reserve_process_resources(running_key, 3, 1, False)
            self.register(running_key, proc)

            utils.release_process_job(running_key, replacement)
            self.claim(next_key)
            with self.assertRaises(utils.ResourceAdmissionError):
                utils.reserve_process_resources(next_key, 1, 1, False)

            utils.release_process_job(running_key, proc)
            self.assertEqual(utils.reserve_process_resources(
                next_key, 3, 1, False), "Reserved 3 CPU slots.")

    def test_natural_completion_releases_resources_for_reuse(self):
        finished_key = self.key("natural-a")
        next_key = self.key("natural-b")
        proc = _CompletedProcess(0)
        state = utils.ProcessStateDict()
        with mock.patch.object(utils, "get_mdrun_cpu_capacity", return_value=2):
            self.claim(finished_key)
            utils.reserve_process_resources(finished_key, 2, 1, False)
            self.register(finished_key, proc)
            utils.set_process_running(
                state, proc, finished_key, "NVT equilibration",
                os.path.dirname(finished_key))

            utils.watch_process(proc, state, finished_key)

            self.assertFalse(state["running"])
            self.assertIsNone(utils.get_process_resource_summary(finished_key))
            self.claim(next_key)
            utils.reserve_process_resources(next_key, 2, 1, False)

    def test_exited_mpi_leader_retains_exact_key_and_directory_leases(self):
        key = self.key("mpi-descendants")
        proc = _ExitedGroupLeader()
        self.claim(key)
        utils.reserve_process_resources(key, 1, 1, False)
        self.register(key, proc)

        with mock.patch.object(
                utils, "_process_group_has_live_members", return_value=True):
            claimed, active = utils.reserve_process_job(key)
            self.assertFalse(claimed)
            self.assertIs(active, proc)
            with self.assertRaises(utils.WorkingDirectoryBusyError):
                with utils.reserve_working_directory_maintenance(
                        os.path.dirname(key)):
                    self.fail("live MPI descendants admitted a maintenance lease")
            with self.assertRaises(utils.WorkingDirectoryBusyError):
                with utils.reserve_working_directory_read(os.path.dirname(key)):
                    self.fail("live MPI descendants admitted a read lease")
            self.assertEqual(utils.get_process_resource_summary(key),
                             "Reserved 1 CPU slot.")

        with mock.patch.object(
                utils, "_process_group_has_live_members", return_value=False):
            claimed, active = utils.reserve_process_job(key)
        self.assertTrue(claimed)
        self.assertIsNone(active)
        self.assertIsNone(utils.get_process_resource_summary(key))

    def test_shutdown_releases_all_resources_and_stops_live_processes(self):
        running_key = self.key("shutdown-a")
        launching_key = self.key("shutdown-b")
        reuse_key = self.key("shutdown-reuse")
        proc = _LiveProcess()
        with mock.patch.object(utils, "get_mdrun_cpu_capacity", return_value=2):
            self.claim(running_key)
            utils.reserve_process_resources(running_key, 1, 1, True)
            self.register(running_key, proc)
            self.claim(launching_key)
            utils.reserve_process_resources(launching_key, 1, 1, False)

            self.assertEqual(utils.stop_all_registered_processes(timeout=0.01), 1)

            self.assertEqual(proc.terminate_calls, 1)
            self.assertIsNone(utils.get_process_resource_summary(running_key))
            self.assertIsNone(utils.get_process_resource_summary(launching_key))
            self.claim(reuse_key)
            utils.reserve_process_resources(reuse_key, 2, 1, True)


class CallbackResourceAdmissionTests(WorkingDirectoryTestCase):
    """Every launch callback must hold and release the shared admission."""

    def new_working_directory(self, suffix: str) -> str:
        directory = tempfile.mkdtemp(prefix=f"_resource_{suffix}_", dir="data")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return directory

    def reserve_existing(self, directory: str, cpu_slots: int,
                         use_gpu: bool = False) -> tuple[str, _LiveProcess]:
        key = utils.get_process_job_key(directory, "existing")
        claimed, active = utils.reserve_process_job(key)
        self.assertTrue(claimed)
        self.assertIsNone(active)
        utils.reserve_process_resources(key, cpu_slots, 1, use_gpu)
        proc = _LiveProcess()
        utils.register_process_job(key, proc)
        self.addCleanup(utils.release_process_job, key, proc)
        return key, proc

    def test_sync_minimization_holds_cpu_admission_around_the_command(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(utils, "get_mdrun_cpu_capacity",
                                      return_value=2):
                blocked_directory = self.new_working_directory(module.__name__)
                blocked_key = utils.get_process_job_key(
                    blocked_directory, "other")

                def run_while_reserved(*_args, **_kwargs):
                    job_key = utils.get_process_job_key(
                        self.working_directory_path, "em")
                    self.assertEqual(utils.get_process_resource_summary(job_key),
                                     "Reserved 2 CPU slots.")
                    claimed, active = utils.reserve_process_job(blocked_key)
                    self.assertTrue(claimed)
                    self.assertIsNone(active)
                    try:
                        with self.assertRaisesRegex(
                                utils.ResourceAdmissionError, "only 0 remain"):
                            utils.reserve_process_resources(
                                blocked_key, 1, 1, False)
                    finally:
                        utils.release_process_job(blocked_key)

                with mock.patch.object(
                        module, "run_checked_command",
                        side_effect=run_while_reserved) as runner:
                    _, status = module.on_run_energy_minimization(
                        self.working_directory_path, "em.tpr", 1, 2, True)

                runner.assert_called_once()
                self.assertIn("color:green", status)
                self.assertIsNone(utils.get_process_resource_summary(
                    utils.get_process_job_key(
                        self.working_directory_path, "em")))

    def test_mdrun_launch_failure_releases_cpu_and_output_reservations(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(utils, "get_mdrun_cpu_capacity",
                                      return_value=2), \
                    mock.patch.object(module.subprocess, "Popen",
                                      side_effect=OSError("cannot launch mdrun")) \
                    as launch:
                state = utils.ProcessStateDict()
                _, status, returned_state, button = \
                    module.on_run_nvt_equilibration(
                        self.working_directory_path, "nvt.tpr", 1, 2,
                        False, state)

                launch.assert_called_once()
                self.assertIn("color:red", status)
                self.assertIn("cannot launch mdrun", self.plain_text(status))
                self.assertFalse(returned_state["running"])
                self.assertEqual(button["value"], "Start")
                failed_key = utils.get_process_job_key(
                    self.working_directory_path, "nvt")
                self.assertIsNone(utils.get_process_resource_summary(failed_key))
                claimed, active = utils.reserve_process_job(failed_key)
                self.assertTrue(claimed)
                self.assertIsNone(active)
                utils.reserve_process_resources(failed_key, 2, 1, False)
                utils.release_process_job(failed_key)

    def test_post_popen_activation_failure_stops_child_and_releases_admission(self):
        for module in (protein_workflow, complex_workflow):
            proc = _LiveProcess()
            with self.subTest(module=module.__name__), \
                    mock.patch.object(utils, "get_mdrun_cpu_capacity",
                                      return_value=2), \
                    mock.patch.object(module.subprocess, "Popen",
                                      return_value=proc), \
                    mock.patch.object(
                        module.threading, "Thread",
                        side_effect=RuntimeError("watcher could not start")):
                state = utils.ProcessStateDict()
                _, status, returned_state, button = \
                    module.on_run_nvt_equilibration(
                        self.working_directory_path, "activation.tpr", 1, 2,
                        False, state)

                self.assertIn("color:red", status)
                self.assertIn("watcher could not start", self.plain_text(status))
                self.assertEqual(proc.terminate_calls, 1)
                self.assertFalse(returned_state["running"])
                self.assertEqual(button["value"], "Start")
                failed_key = utils.get_process_job_key(
                    self.working_directory_path, "activation")
                self.assertIsNone(utils.get_process_resource_summary(failed_key))
                claimed, active = utils.reserve_process_job(failed_key)
                self.assertTrue(claimed)
                self.assertIsNone(active)
                utils.release_process_job(failed_key)

    def test_mmpbsa_prelaunch_failure_releases_its_cpu_admission(self):
        with mock.patch.object(utils, "get_mdrun_cpu_capacity", return_value=3), \
                mock.patch.object(
                    complex_workflow, "get_gmx_mmpbsa_executable",
                    return_value="/opt/gmxMMPBSA/bin/gmx_MMPBSA"), \
                mock.patch.object(
                    complex_workflow, "_build_mmpbsa_index",
                    side_effect=RuntimeError("index construction failed")), \
                mock.patch.object(complex_workflow.subprocess, "Popen") as launch:
            state = utils.ProcessStateDict()
            _, status, returned_state, button = complex_workflow.on_run_mmpbsa(
                self.working_directory_path, "md.tpr", "md.xtc", "topol.top",
                "mmpbsa.in", "mmpbsa_index.ndx", "protein", "resname LIG",
                3, state)

            launch.assert_not_called()
            self.assertIn("color:red", status)
            self.assertIn("index construction failed", self.plain_text(status))
            self.assertFalse(returned_state["running"])
            self.assertEqual(button["value"], "Start")
            key = utils.get_process_job_key(
                self.working_directory_path,
                complex_workflow.MMPBSA_RESULTS_FILE_NAME)
            self.assertIsNone(utils.get_process_resource_summary(key))
            claimed, active = utils.reserve_process_job(key)
            self.assertTrue(claimed)
            self.assertIsNone(active)
            utils.release_process_job(key)

    def test_gpu_denial_is_red_and_stopped_job_can_be_reused(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(utils, "get_mdrun_cpu_capacity",
                                      return_value=4), \
                    mock.patch.object(module.threading, "Thread") as thread, \
                    mock.patch.object(module.subprocess, "Popen") as launch:
                first_proc, reused_proc = _LiveProcess(), _LiveProcess()
                launch.side_effect = [first_proc, reused_proc]
                first_state = utils.ProcessStateDict()
                second_state = utils.ProcessStateDict()
                second_directory = self.new_working_directory(
                    "gpu_" + module.__name__)

                _, first_status, _, _ = module.on_run_nvt_equilibration(
                    self.working_directory_path, "nvt.tpr", 1, 2,
                    True, first_state)
                _, denied_status, _, denied_button = \
                    module.on_run_nvt_equilibration(
                        second_directory, "nvt.tpr", 1, 1,
                        True, second_state)

                self.assertIn("color:orange", first_status)
                self.assertIn("Reserved 2 CPU slots and exclusive GPU use",
                              self.plain_text(first_status))
                self.assertIn("color:red", denied_status)
                self.assertIn("GPU is already reserved",
                              self.plain_text(denied_status))
                self.assertIn("clear Use GPU", self.plain_text(denied_status))
                self.assertEqual(denied_button["value"], "Start")
                self.assertEqual(launch.call_count, 1)

                with mock.patch.object(module, "stop_process_gracefully") as stop:
                    module.on_run_nvt_equilibration(
                        self.working_directory_path, "nvt.tpr", 1, 2,
                        True, first_state)
                stop.assert_called_once_with(first_proc)

                _, reused_status, reused_state, _ = \
                    module.on_run_nvt_equilibration(
                        second_directory, "nvt.tpr", 1, 1,
                        True, second_state)
                self.assertIn("color:orange", reused_status)
                self.assertTrue(reused_state["running"])
                self.assertEqual(launch.call_count, 2)
                reused_key = utils.get_process_job_key(second_directory, "nvt")
                utils.release_process_job(reused_key, reused_proc)
                utils.clear_process_state(second_state)
                thread.assert_called()

    def test_attached_session_does_not_call_the_resource_reserver_again(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(utils, "get_mdrun_cpu_capacity",
                                      return_value=4), \
                    mock.patch.object(module.threading, "Thread"), \
                    mock.patch.object(module.subprocess, "Popen",
                                      return_value=_LiveProcess()) as launch, \
                    mock.patch.object(
                        module, "reserve_process_resources",
                        wraps=module.reserve_process_resources) as reserve:
                original_state = utils.ProcessStateDict()
                attached_state = utils.ProcessStateDict()
                module.on_run_nvt_equilibration(
                    self.working_directory_path, "attach.tpr", 1, 2,
                    False, original_state)
                _, status, returned_state, button = \
                    module.on_run_nvt_equilibration(
                        self.working_directory_path, "attach.tpr", 2, 2,
                        True, attached_state)

                launch.assert_called_once()
                reserve.assert_called_once()
                self.assertIn("color:orange", status)
                self.assertIn("now attached", self.plain_text(status))
                self.assertTrue(returned_state["running"])
                self.assertEqual(button["value"], "Stop")
                key = utils.get_process_job_key(
                    self.working_directory_path, "attach")
                proc = original_state["proc"]
                utils.release_process_job(key, proc)
                utils.clear_process_state(original_state)
                utils.clear_process_state(attached_state)

    def test_mmpbsa_processes_share_the_cpu_budget_and_error_names_its_control(self):
        other_directory = self.new_working_directory("mmpbsa_existing")
        with mock.patch.object(utils, "get_mdrun_cpu_capacity", return_value=4):
            self.reserve_existing(other_directory, 2)
            state = utils.ProcessStateDict()
            with mock.patch.object(
                    complex_workflow, "get_default_cpu_count", return_value=4), \
                    mock.patch.object(
                        complex_workflow, "get_gmx_mmpbsa_executable",
                        return_value="/opt/gmxMMPBSA/bin/gmx_MMPBSA"), \
                    mock.patch.object(
                        complex_workflow, "_build_mmpbsa_index") as build, \
                    mock.patch.object(
                        complex_workflow.subprocess, "Popen") as launch:
                _, status, returned_state, button = \
                    complex_workflow.on_run_mmpbsa(
                        self.working_directory_path, "md.tpr", "md.xtc",
                        "topol.top", "mmpbsa.in", "mmpbsa_index.ndx",
                        "protein", "resname LIG", 3, state)

            build.assert_not_called()
            launch.assert_not_called()
            self.assertIn("color:red", status)
            text = self.plain_text(status)
            self.assertIn("3 CPU slots (MM-PBSA processes)", text)
            self.assertIn("only 2 remain", text)
            self.assertIn("reduce MM-PBSA processes", text)
            self.assertFalse(returned_state["running"])
            self.assertEqual(button["value"], "Start")


if __name__ == "__main__":
    unittest.main()

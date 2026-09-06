"""Server startup, shutdown, upload-limit, and port-selection regressions."""

import asyncio
import inspect
import unittest
from unittest import mock

import webui


class WebUiLifecycleTests(unittest.TestCase):
    def test_gradio_mount_retains_application_lifespan(self):
        mounted_context = getattr(
            webui.app.router.lifespan_context,
            "__wrapped__",
            webui.app.router.lifespan_context,
        )
        self.assertIs(
            inspect.getclosurevars(mounted_context).nonlocals.get("old_lifespan"),
            webui.application_lifespan,
        )

    def test_application_lifespan_runs_cleanup_and_stops_registered_processes(self):
        async def exercise_lifespan():
            with (
                mock.patch.object(webui, "cleanup_stale_static_assets") as cleanup,
                mock.patch.object(
                    webui.utils, "stop_all_registered_processes") as stop_processes,
            ):
                # Exercise the application's own context directly. Gradio wraps
                # this context when mounting; entering the combined context in a
                # unit test also starts persistent queue workers owned by Gradio.
                async with webui.application_lifespan(webui.app):
                    cleanup.assert_called_once_with(
                        webui.STATIC_ASSET_MAX_AGE_SECONDS)
                    stop_processes.assert_not_called()

                stop_processes.assert_called_once_with(
                    timeout=webui.PROCESS_SHUTDOWN_TIMEOUT_SECONDS)

        asyncio.run(exercise_lifespan())

    def test_gradio_upload_size_is_bounded(self):
        self.assertEqual(webui.blocks.max_file_size, webui.MAX_UPLOAD_SIZE_BYTES)
        self.assertEqual(webui.MAX_UPLOAD_SIZE_BYTES, 100 * 1024 * 1024)

    def test_port_scan_advances_and_is_bounded(self):
        socket_instance = mock.MagicMock()
        socket_instance.__enter__.return_value = socket_instance
        socket_instance.bind.side_effect = [OSError("busy"), None]
        with mock.patch.object(webui.socket, "socket", return_value=socket_instance) as factory:
            selected = webui.find_available_port(8000, max_attempts=2)

        self.assertEqual(selected, 8001)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(
            socket_instance.bind.call_args_list,
            [mock.call((webui.SERVER_HOST, 8000)), mock.call((webui.SERVER_HOST, 8001))],
        )

        always_busy = mock.MagicMock()
        always_busy.__enter__.return_value = always_busy
        always_busy.bind.side_effect = OSError("busy")
        with mock.patch.object(webui.socket, "socket", return_value=always_busy):
            with self.assertRaisesRegex(RuntimeError, "8000 through 8001"):
                webui.find_available_port(8000, max_attempts=2)
        self.assertEqual(always_busy.bind.call_count, 2)

    def test_port_scan_validates_arguments_and_does_not_wrap_past_65535(self):
        for start_port in (0, 65536, True, 8000.0):
            with self.subTest(start_port=start_port), self.assertRaises(ValueError):
                webui.find_available_port(start_port)
        for attempts in (0, -1, True, 2.0):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                webui.find_available_port(8000, attempts)

        busy = mock.MagicMock()
        busy.__enter__.return_value = busy
        busy.bind.side_effect = OSError("busy")
        with mock.patch.object(webui.socket, "socket", return_value=busy):
            with self.assertRaisesRegex(RuntimeError, "65535 through 65535"):
                webui.find_available_port(65535, max_attempts=10)
        busy.bind.assert_called_once_with((webui.SERVER_HOST, 65535))

        with mock.patch.object(
            webui.socket, "socket", side_effect=PermissionError("not permitted")
        ) as factory:
            with self.assertRaisesRegex(RuntimeError, "9000 through 9002"):
                webui.find_available_port(9000, max_attempts=3)
        self.assertEqual(factory.call_count, 3)


if __name__ == "__main__":
    unittest.main()

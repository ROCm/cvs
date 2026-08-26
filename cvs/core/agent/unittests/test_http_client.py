'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/agent/http_client.py: ParallelHTTPClient's run_command/health/shutdown
# fan-out, host_args/stop_on_errors semantics, exception classification, and session lifecycle.

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from cvs.core.agent import messages
from cvs.core.agent.http_client import (
    HostOutput,
    HTTPConnectionError,
    HTTPProtocolError,
    ParallelHTTPClient,
    ParallelHTTPClientError,
)
from cvs.core.run_layout import RunLayout

TOKEN = "test-token-123"


def _exec_handler(request: httpx.Request) -> httpx.Response:
    '''Default /v1/exec responder: echoes the requested cmd back as stdout, exit_code 0.'''
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "exit_code": 0,
            "stdout": [body["cmd"]],
            "stderr": [],
            "stdout_path": None,
            "stderr_path": None,
            "truncated": False,
            "timed_out": False,
        },
    )


class HttpClientTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # run_command's cwd/out_path derive from RunLayout; point it at a throwaway tempdir so
        # tests don't create real cvs_runs/ directories under the repo (see test_utils_lib.py's
        # TestResolveRunDirPlaceholder for the same pattern).
        RunLayout._reset()
        self.addCleanup(RunLayout._reset)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        RunLayout.get(self.tmp.name)

    def _make_client(self, agent_urls: dict[str, str], handler, token: str = TOKEN, **kwargs) -> ParallelHTTPClient:
        client = ParallelHTTPClient(agent_urls, token, transport=httpx.MockTransport(handler), **kwargs)
        self.addAsyncCleanup(client.destroy)
        return client


class TestRunCommand(HttpClientTestBase):
    async def test_returns_host_output_per_host(self):
        client = self._make_client({"h1": "http://h1", "h2": "http://h2"}, _exec_handler)
        outputs = await client.run_command("echo hi")
        self.assertEqual({o.host for o in outputs}, {"h1", "h2"})
        for output in outputs:
            self.assertEqual(output.stdout, ["echo hi"])
            self.assertEqual(output.stderr, [])
            self.assertEqual(output.exit_code, 0)
            self.assertIsNone(output.exception)

    async def test_sends_bearer_auth_header(self):
        seen_headers = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(request.headers.get("authorization"))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler, token="secret-abc")
        await client.run_command("true")
        self.assertEqual(seen_headers, [f"{messages.AUTH_SCHEME} secret-abc"])

    async def test_posts_to_exec_path_on_each_hosts_url(self):
        seen_urls = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_urls.append(str(request.url))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://host-one", "h2": "http://host-two"}, handler)
        await client.run_command("true")
        self.assertEqual(
            sorted(seen_urls), sorted([f"http://host-one{messages.EXEC_PATH}", f"http://host-two{messages.EXEC_PATH}"])
        )

    async def test_host_args_substitutes_a_different_command_per_host(self):
        client = self._make_client({"h1": "http://h1", "h2": "http://h2"}, _exec_handler)
        outputs = await client.run_command("echo %s", host_args=["one", "two"])
        by_host = {o.host: o.stdout for o in outputs}
        self.assertEqual(by_host, {"h1": ["echo one"], "h2": ["echo two"]})

    async def test_host_args_length_mismatch_raises_value_error(self):
        client = self._make_client({"h1": "http://h1", "h2": "http://h2"}, _exec_handler)
        with self.assertRaises(ValueError):
            await client.run_command("echo %s", host_args=["only-one"])

    async def test_stop_on_errors_true_raises_when_a_host_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "bad" in str(request.url):
                return httpx.Response(500, text="boom")
            return _exec_handler(request)

        client = self._make_client({"good": "http://good", "bad": "http://bad"}, handler)
        with self.assertRaises(ParallelHTTPClientError):
            await client.run_command("true", stop_on_errors=True)

    async def test_stop_on_errors_false_returns_partial_results(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "bad" in str(request.url):
                return httpx.Response(500, text="boom")
            return _exec_handler(request)

        client = self._make_client({"good": "http://good", "bad": "http://bad"}, handler)
        outputs = await client.run_command("true", stop_on_errors=False)
        by_host = {o.host: o for o in outputs}
        self.assertIsNone(by_host["good"].exception)
        self.assertEqual(by_host["good"].exit_code, 0)
        self.assertIsInstance(by_host["bad"].exception, HTTPProtocolError)

    async def test_nonzero_remote_exit_code_is_not_a_stop_on_errors_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "exit_code": 1,
                    "stdout": [],
                    "stderr": ["failed"],
                    "stdout_path": None,
                    "stderr_path": None,
                    "truncated": False,
                    "timed_out": False,
                },
            )

        client = self._make_client({"h1": "http://h1"}, handler)
        outputs = await client.run_command("false", stop_on_errors=True)
        self.assertEqual(outputs[0].exit_code, 1)
        self.assertIsNone(outputs[0].exception)

    async def test_read_timeout_is_rounded_to_int_for_the_wire_request(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.run_command("true", read_timeout=2.7)
        self.assertEqual(seen_requests[0].timeout, 3)

    async def test_inactivity_timeout_is_rounded_to_int_for_the_wire_request(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.run_command("true", inactivity_timeout=4.4)
        self.assertEqual(seen_requests[0].inactivity_timeout, 4)

    async def test_inactivity_timeout_defaults_to_none(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.run_command("true")
        self.assertIsNone(seen_requests[0].inactivity_timeout)

    async def test_env_is_passed_through_to_the_exec_request(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.run_command("true", env={"FOO": "bar"})
        self.assertEqual(seen_requests[0].env, {"FOO": "bar"})

    async def test_env_defaults_to_empty_dict(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.run_command("true")
        self.assertEqual(seen_requests[0].env, {})

    async def test_cwd_defaults_to_the_run_layout_run_dir_not_a_local_path(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.run_command("true")
        self.assertEqual(seen_requests[0].cwd, RunLayout.get().run_dir)

    async def test_client_is_reused_across_calls(self):
        client = self._make_client({"h1": "http://h1"}, _exec_handler)
        await client.run_command("true")
        first_client = client._client
        await client.run_command("true")
        self.assertIs(client._client, first_client)


def _file_mode_handler(request: httpx.Request) -> httpx.Response:
    '''Simulates the agent's FILE output_mode: writes the full output to out_path on the (here,
    tempdir-backed) shared FS and returns only a short tail preview inline, like http_agent.py does.'''
    body = json.loads(request.content)
    out_dir = Path(body["out_path"])
    stdout_path = out_dir / f"{body['cmd_id']}.stdout"
    stderr_path = out_dir / f"{body['cmd_id']}.stderr"
    full_stdout = "\n".join(f"line{i}" for i in range(20))
    stdout_path.write_text(full_stdout)
    stderr_path.write_text("err-line")
    return httpx.Response(
        200,
        json={
            "exit_code": 0,
            "stdout": full_stdout.splitlines()[-2:],
            "stderr": ["err-line"],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "truncated": None,
            "timed_out": False,
        },
    )


class TestOutputMode(HttpClientTestBase):
    async def test_defaults_to_inline_and_does_not_set_out_path(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return _exec_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.run_command("true")
        self.assertEqual(seen_requests[0].output_mode, messages.ExecOutputMode.INLINE)
        self.assertIsNone(seen_requests[0].out_path)

    async def test_file_mode_sets_out_path_under_the_run_layout_run_dir(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return _file_mode_handler(request)

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.run_command("true", output_mode=messages.ExecOutputMode.FILE)
        self.assertEqual(seen_requests[0].out_path, RunLayout.get().run_dir / "exec_output")
        self.assertTrue(seen_requests[0].out_path.is_dir())

    async def test_file_mode_returns_full_output_not_just_the_inline_preview(self):
        client = self._make_client({"h1": "http://h1"}, _file_mode_handler)
        outputs = await client.run_command("true", output_mode=messages.ExecOutputMode.FILE)
        self.assertEqual(outputs[0].stdout, [f"line{i}" for i in range(20)])
        self.assertEqual(outputs[0].stderr, ["err-line"])

    async def test_exit_code_only_mode_is_sent_through(self):
        seen_requests: list[messages.ExecRequest] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_requests.append(messages.parse_message(messages.ExecRequest, request.content.decode()))
            return httpx.Response(
                200,
                json={
                    "exit_code": 0,
                    "stdout": None,
                    "stderr": None,
                    "stdout_path": None,
                    "stderr_path": None,
                    "truncated": None,
                    "timed_out": False,
                },
            )

        client = self._make_client({"h1": "http://h1"}, handler)
        outputs = await client.run_command("true", output_mode=messages.ExecOutputMode.EXIT_CODE_ONLY)
        self.assertEqual(seen_requests[0].output_mode, messages.ExecOutputMode.EXIT_CODE_ONLY)
        self.assertEqual(outputs[0].stdout, [])
        self.assertEqual(outputs[0].stderr, [])
        self.assertEqual(outputs[0].exit_code, 0)


class TestExceptionClassification(HttpClientTestBase):
    async def test_connection_failure_is_classified_as_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = self._make_client({"h1": "http://h1"}, handler)
        outputs = await client.run_command("true", stop_on_errors=False)
        self.assertIsInstance(outputs[0].exception, HTTPConnectionError)

    async def test_read_timeout_is_classified_as_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client = self._make_client({"h1": "http://h1"}, handler)
        outputs = await client.run_command("true", stop_on_errors=False)
        self.assertIsInstance(outputs[0].exception, HTTPConnectionError)

    async def test_bad_http_status_is_classified_as_protocol_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        client = self._make_client({"h1": "http://h1"}, handler)
        outputs = await client.run_command("true", stop_on_errors=False)
        self.assertIsInstance(outputs[0].exception, HTTPProtocolError)

    async def test_unparseable_response_body_is_classified_as_protocol_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")

        client = self._make_client({"h1": "http://h1"}, handler)
        outputs = await client.run_command("true", stop_on_errors=False)
        self.assertIsInstance(outputs[0].exception, HTTPProtocolError)


class TestHealth(HttpClientTestBase):
    async def test_all_hosts_healthy(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        client = self._make_client({"h1": "http://h1", "h2": "http://h2"}, handler)
        self.assertEqual(await client.health(), {"h1": True, "h2": True})

    async def test_unreachable_host_reported_as_unhealthy_without_raising(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "down" in str(request.url):
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json={"ok": True})

        client = self._make_client({"up": "http://up", "down": "http://down"}, handler)
        self.assertEqual(await client.health(), {"up": True, "down": False})

    async def test_hits_health_path(self):
        seen_paths = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            return httpx.Response(200, json={"ok": True})

        client = self._make_client({"h1": "http://h1"}, handler)
        await client.health()
        self.assertEqual(seen_paths, [messages.HEALTH_PATH])


class TestShutdown(HttpClientTestBase):
    async def test_hits_shutdown_path_on_every_host(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path, str(request.url)))
            return httpx.Response(200, json={"ok": True})

        client = self._make_client({"h1": "http://h1", "h2": "http://h2"}, handler)
        result = await client.shutdown()
        self.assertEqual(result, {"h1": True, "h2": True})
        self.assertEqual({(m, p) for m, p, _ in seen}, {("POST", messages.SHUTDOWN_PATH)})
        self.assertEqual(
            {url for _, _, url in seen}, {f"http://h1{messages.SHUTDOWN_PATH}", f"http://h2{messages.SHUTDOWN_PATH}"}
        )

    async def test_default_is_best_effort_and_does_not_raise_on_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "bad" in str(request.url):
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"ok": True})

        client = self._make_client({"good": "http://good", "bad": "http://bad"}, handler)
        result = await client.shutdown()
        self.assertEqual(result, {"good": True, "bad": False})

    async def test_stop_on_errors_true_raises_when_a_host_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "bad" in str(request.url):
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"ok": True})

        client = self._make_client({"good": "http://good", "bad": "http://bad"}, handler)
        with self.assertRaises(ParallelHTTPClientError):
            await client.shutdown(stop_on_errors=True)


class TestRebuildAndDestroy(HttpClientTestBase):
    async def test_rebuild_replaces_host_map(self):
        client = self._make_client({"h1": "http://h1", "h2": "http://h2"}, _exec_handler)
        client.rebuild({"h3": "http://h3"})
        outputs = await client.run_command("true")
        self.assertEqual([o.host for o in outputs], ["h3"])

    async def test_destroy_closes_client_and_allows_lazy_recreation(self):
        client = self._make_client({"h1": "http://h1"}, _exec_handler)
        await client.run_command("true")
        self.assertIsNotNone(client._client)
        await client.destroy()
        self.assertIsNone(client._client)
        # a call after destroy() lazily recreates the client rather than failing
        outputs = await client.run_command("true")
        self.assertEqual(outputs[0].exit_code, 0)

    async def test_async_context_manager_destroys_on_exit(self):
        client = ParallelHTTPClient({"h1": "http://h1"}, TOKEN, transport=httpx.MockTransport(_exec_handler))
        async with client as ctx_client:
            self.assertIs(ctx_client, client)
            await client.run_command("true")
            self.assertIsNotNone(client._client)
        self.assertIsNone(client._client)


class TestHostOutput(unittest.TestCase):
    def test_is_a_plain_dataclass_not_pssh_output(self):
        output = HostOutput(host="h1", stdout=["a"], stderr=["b"], exit_code=0, exception=None)
        self.assertEqual(output.host, "h1")
        self.assertEqual(output.stdout, ["a"])
        self.assertEqual(output.stderr, ["b"])
        self.assertEqual(output.exit_code, 0)
        self.assertIsNone(output.exception)


if __name__ == "__main__":
    unittest.main()

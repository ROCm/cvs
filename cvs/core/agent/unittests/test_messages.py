'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/agent/messages.py: request/response schema validation,
# the INLINE/FILE/EXIT_CODE_ONLY ExecRequest/ExecResponse contract, and the
# parse_message/MessageParseError deserialization wrapper.

import unittest
from pathlib import Path

from pydantic import ValidationError

from cvs.core.agent.messages import (
    ErrorResponse,
    ExecOutputMode,
    ExecRequest,
    ExecResponse,
    MessageParseError,
    RegisterRequest,
    RegisterResponse,
    ShutdownRequest,
    parse_message,
)


class TestRegisterRequest(unittest.TestCase):
    def test_accepts_valid_rank_and_port(self):
        req = RegisterRequest(rank=1, hostname="node1", port=8080)
        self.assertEqual(req.rank, 1)
        self.assertEqual(req.hostname, "node1")
        self.assertEqual(req.port, 8080)

    def test_rejects_negative_rank(self):
        with self.assertRaises(ValidationError):
            RegisterRequest(rank=-1, hostname="node1", port=8080)

    def test_rejects_out_of_range_port(self):
        with self.assertRaises(ValidationError):
            RegisterRequest(rank=0, hostname="node1", port=70000)


class TestExecRequest(unittest.TestCase):
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            cmd="echo hi",
            env={},
            cwd=Path("/tmp"),
            timeout=None,
            inactivity_timeout=None,
            cmd_id="abc123",
            out_path=None,
            output_mode=ExecOutputMode.INLINE,
        )
        kwargs.update(overrides)
        return kwargs

    def test_timeout_and_inactivity_timeout_accept_none(self):
        req = ExecRequest(**self._base_kwargs())
        self.assertIsNone(req.timeout)
        self.assertIsNone(req.inactivity_timeout)

    def test_timeout_and_inactivity_timeout_accept_int(self):
        req = ExecRequest(**self._base_kwargs(timeout=30, inactivity_timeout=10))
        self.assertEqual(req.timeout, 30)
        self.assertEqual(req.inactivity_timeout, 10)

    def test_inline_mode_does_not_require_out_path(self):
        req = ExecRequest(**self._base_kwargs(output_mode=ExecOutputMode.INLINE))
        self.assertIsNone(req.out_path)

    def test_exit_code_only_mode_does_not_require_out_path(self):
        req = ExecRequest(**self._base_kwargs(output_mode=ExecOutputMode.EXIT_CODE_ONLY))
        self.assertIsNone(req.out_path)

    def test_file_mode_requires_out_path(self):
        with self.assertRaises(ValidationError):
            ExecRequest(**self._base_kwargs(output_mode=ExecOutputMode.FILE, out_path=None))

    def test_file_mode_accepts_out_path(self):
        req = ExecRequest(**self._base_kwargs(output_mode=ExecOutputMode.FILE, out_path=Path("/tmp/out")))
        self.assertEqual(req.out_path, Path("/tmp/out"))

    def test_round_trip_json(self):
        req = ExecRequest(**self._base_kwargs(timeout=30))
        restored = parse_message(ExecRequest, req.model_dump_json())
        self.assertEqual(req, restored)


class TestExecResponse(unittest.TestCase):
    def test_inline_mode_response(self):
        resp = ExecResponse(
            exit_code=0,
            stdout=["hi"],
            stderr=[],
            stdout_path=None,
            stderr_path=None,
            truncated=False,
        )
        self.assertEqual(resp.stdout, ["hi"])
        self.assertIsNone(resp.stdout_path)

    def test_file_mode_response(self):
        resp = ExecResponse(
            exit_code=0,
            stdout=["...tail..."],
            stderr=[],
            stdout_path=Path("/tmp/out/abc123.stdout"),
            stderr_path=Path("/tmp/out/abc123.stderr"),
            truncated=None,
        )
        self.assertEqual(resp.stdout_path, Path("/tmp/out/abc123.stdout"))

    def test_exit_code_only_response(self):
        resp = ExecResponse(
            exit_code=1,
            stdout=None,
            stderr=None,
            stdout_path=None,
            stderr_path=None,
            truncated=None,
        )
        self.assertEqual(resp.exit_code, 1)
        self.assertIsNone(resp.stdout)


class TestShutdownRequest(unittest.TestCase):
    def test_takes_no_fields(self):
        req = ShutdownRequest()
        self.assertEqual(req.model_dump(), {})


class TestParseMessage(unittest.TestCase):
    def test_returns_validated_model_on_success(self):
        raw = RegisterResponse(ok=True).model_dump_json()
        result = parse_message(RegisterResponse, raw)
        self.assertIsInstance(result, RegisterResponse)
        self.assertTrue(result.ok)

    def test_raises_message_parse_error_on_bad_input(self):
        with self.assertRaises(MessageParseError):
            parse_message(RegisterRequest, '{"rank": "not-an-int"}')

    def test_preserves_original_validation_error_as_cause(self):
        try:
            parse_message(ErrorResponse, "{}")
        except MessageParseError as exc:
            self.assertIsInstance(exc.__cause__, ValidationError)
        else:
            self.fail("MessageParseError was not raised")


if __name__ == "__main__":
    unittest.main()

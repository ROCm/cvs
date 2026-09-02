'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import unittest

from cvs.lib.utils.model_query_lib import OpenAIProbe


def _chat_body(*, content="", reasoning_content="", model="test/model"):
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning_content,
                }
            }
        ],
    }


class TestOpenAIProbeReasoningModels(unittest.TestCase):
    def test_chat_accepts_reasoning_content_when_content_empty(self):
        results = {
            "model_endpoint": (200, {"data": [{"id": "deepseek-ai/DeepSeek-R1-0528"}]}),
            "chat_completion_endpoint": (
                200,
                _chat_body(
                    model="deepseek-ai/DeepSeek-R1-0528",
                    reasoning_content="Hmm, the user just",
                ),
            ),
            "completion_endpoint": (200, {"model": "m", "choices": [{"text": " Paris"}]}),
            "structured_output_book": (
                200,
                _chat_body(
                    model="deepseek-ai/DeepSeek-R1-0528",
                    reasoning_content="Planning the JSON response...",
                ),
            ),
        }
        ok, err = OpenAIProbe.check_results(results, port=8000)
        self.assertTrue(ok, err)

    def test_chat_still_requires_visible_text(self):
        results = {
            "model_endpoint": (200, {"data": [{"id": "test/model"}]}),
            "chat_completion_endpoint": (200, _chat_body()),
            "completion_endpoint": (200, {"model": "m", "choices": [{"text": "x"}]}),
            "structured_output_book": (
                200,
                _chat_body(content='{"title":"t","author":"a","year":2000,"genre":"g"}'),
            ),
        }
        ok, err = OpenAIProbe.check_results(results, port=8000)
        self.assertFalse(ok)
        self.assertIn("empty assistant content", err or "")

    def test_structured_book_still_validates_json_for_non_reasoning_models(self):
        results = {
            "model_endpoint": (200, {"data": [{"id": "meta/llama"}]}),
            "chat_completion_endpoint": (200, _chat_body(content="OK")),
            "completion_endpoint": (200, {"model": "m", "choices": [{"text": "x"}]}),
            "structured_output_book": (
                200,
                _chat_body(model="meta/llama", reasoning_content="thinking only"),
            ),
        }
        ok, err = OpenAIProbe.check_results(results, port=8000)
        self.assertFalse(ok)
        self.assertIn("empty assistant content", err or "")

    def test_structured_book_accepts_markdown_fenced_json(self):
        fenced = (
            '```json\n{"title": "To Kill a Mockingbird", '
            '"author": "Harper Lee", "year": 1960, "genre": "Southern Gothic"}\n```'
        )
        results = {
            "model_endpoint": (200, {"data": [{"id": "deepseek-ai/DeepSeek-R1-0528"}]}),
            "chat_completion_endpoint": (
                200,
                _chat_body(
                    model="deepseek-ai/DeepSeek-R1-0528",
                    reasoning_content="Hmm, the user just",
                ),
            ),
            "completion_endpoint": (200, {"model": "m", "choices": [{"text": " Paris"}]}),
            "structured_output_book": (
                200,
                _chat_body(model="deepseek-ai/DeepSeek-R1-0528", content=fenced),
            ),
        }
        ok, err = OpenAIProbe.check_results(results, port=8000)
        self.assertTrue(ok, err)

    def test_structured_book_accepts_qwen_thinking_content(self):
        results = {
            "model_endpoint": (200, {"data": [{"id": "/data/Qwen3.5-397B-A17B-FP8"}]}),
            "chat_completion_endpoint": (
                200,
                _chat_body(
                    model="/data/Qwen3.5-397B-A17B-FP8",
                    content="Thinking Process:\n\n1. **Analyze",
                ),
            ),
            "completion_endpoint": (200, {"model": "m", "choices": [{"text": " Paris."}]}),
            "structured_output_book": (
                200,
                _chat_body(
                    model="/data/Qwen3.5-397B-A17B-FP8",
                    content="Thinking Process:\n\n1. **Analyze the Request:**",
                ),
            ),
        }
        ok, err = OpenAIProbe.check_results(results, port=8000)
        self.assertTrue(ok, err)

    def test_extract_json_object_finds_object_after_thinking_preamble(self):
        text = (
            "Thinking Process:\n\n"
            'Here is the book: {"title": "1984", "author": "George Orwell", "year": 1949, "genre": "Dystopian"}'
        )
        obj = OpenAIProbe._extract_json_object(text)
        self.assertIsNotNone(obj)
        self.assertEqual(obj["title"], "1984")


if __name__ == "__main__":
    unittest.main()

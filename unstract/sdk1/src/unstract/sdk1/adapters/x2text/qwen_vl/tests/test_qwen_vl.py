"""Unit tests for the Qwen 2.5 VL X2Text adapter.

Mocks all openai calls so no real API credentials or network connectivity are
required.  Tests verify:

- Correct output of ``_split_pages`` for delimiter-delimited and
  delimiter-free text.
- ``process()`` returns a valid ``TextExtractionResult`` with the right
  page count, text, and optional file writes.
- ``test_connection()`` propagates API errors as ``AdapterError``.
- Error cases: missing base URL, rate limit, connection refused, empty
  response, unsupported MIME type fallback.
- API key / base URL resolution from config and environment variables.
- PDF-to-image conversion pipeline.
"""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType
import unittest

# Stub python-magic so importing FileStorage does not depend on libmagic binary
sys.modules.setdefault("magic", ModuleType("magic"))

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

from unstract.sdk1.adapters.exceptions import AdapterError
from unstract.sdk1.adapters.x2text.qwen_vl.src.qwen_vl import (
    QwenVLAdapter,
)
from unstract.sdk1.adapters.x2text.qwen_vl.src.constants import (
    QwenVLDefaults,
    QwenVLMimeTypes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PDF_MAGIC = b"%PDF-1.4\n"


def _make_adapter(extra_config: dict | None = None) -> QwenVLAdapter:
    config: dict = {
        "base_url": "http://localhost:8000/v1",
        "api_key": "test-key",
    }
    if extra_config:
        config.update(extra_config)
    return QwenVLAdapter(settings=config)


def _fake_openai_response(
    text: str, prompt_tokens: int = 100, completion_tokens: int = 200
):
    """Build a fake openai ChatCompletion response namespace."""
    message = SimpleNamespace(content=text, role="assistant")
    choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# _split_pages
# ---------------------------------------------------------------------------

class TestSplitPages(unittest.TestCase):
    """Unit tests for the static ``_split_pages`` helper."""

    def test_single_page_no_delimiter(self):
        raw = "Hello world"
        pages = QwenVLAdapter._split_pages(raw)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[0]["text"], "Hello world")

    def test_blank_text_returns_empty(self):
        pages = QwenVLAdapter._split_pages("   ")
        self.assertEqual(pages, [])

    def test_single_delimiter(self):
        raw = "<<< PAGE_1 >>>\nContent of page one."
        pages = QwenVLAdapter._split_pages(raw)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("Content of page one", pages[0]["text"])

    def test_multi_page(self):
        raw = (
            "<<< PAGE_1 >>>\nFirst page text.\n"
            "<<< PAGE_2 >>>\nSecond page text.\n"
            "<<< PAGE_3 >>>\nThird page text."
        )
        pages = QwenVLAdapter._split_pages(raw)
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[2]["page_number"], 3)
        self.assertIn("Third page", pages[2]["text"])

    def test_delimiter_case_insensitive(self):
        raw = "<<< page_1 >>>\nLower case delimiter."
        pages = QwenVLAdapter._split_pages(raw)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)

    def test_pages_sorted_by_number(self):
        raw = "<<< PAGE_2 >>>\nSecond.\n<<< PAGE_1 >>>\nFirst."
        pages = QwenVLAdapter._split_pages(raw)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[1]["page_number"], 2)

    def test_whitespace_in_delimiter(self):
        raw = "<<<  PAGE_4  >>>\nContent."
        pages = QwenVLAdapter._split_pages(raw)
        self.assertEqual(pages[0]["page_number"], 4)


# ---------------------------------------------------------------------------
# _is_delimiter_only
# ---------------------------------------------------------------------------

class TestIsDelimiterOnly(unittest.TestCase):
    """Unit tests for the ``_is_delimiter_only`` helper."""

    def test_delimiter_only_single(self):
        self.assertTrue(
            QwenVLAdapter._is_delimiter_only("<<< PAGE_1 >>>")
        )

    def test_delimiter_only_multi(self):
        text = "<<< PAGE_1 >>>\n\n<<< PAGE_2 >>>\n\n<<< PAGE_3 >>>"
        self.assertTrue(QwenVLAdapter._is_delimiter_only(text))

    def test_delimiter_with_whitespace_only(self):
        text = "  <<< PAGE_1 >>>  \n  \n  <<< PAGE_2 >>>  "
        self.assertTrue(QwenVLAdapter._is_delimiter_only(text))

    def test_real_content_returns_false(self):
        text = "<<< PAGE_1 >>>\nThis is actual content."
        self.assertFalse(QwenVLAdapter._is_delimiter_only(text))

    def test_no_delimiters_with_text_returns_false(self):
        self.assertFalse(
            QwenVLAdapter._is_delimiter_only("Just some plain text.")
        )

    def test_empty_string_returns_true(self):
        # Edge case: empty string has no content
        self.assertTrue(QwenVLAdapter._is_delimiter_only(""))

    def test_whitespace_only_returns_true(self):
        self.assertTrue(QwenVLAdapter._is_delimiter_only("   \n\n  "))

    def test_case_insensitive(self):
        self.assertTrue(
            QwenVLAdapter._is_delimiter_only("<<< page_1 >>>")
        )


# ---------------------------------------------------------------------------
# MIME type detection
# ---------------------------------------------------------------------------

class TestDetectMimeType(unittest.TestCase):

    def _make_fs(self, mime_return: str = "application/pdf") -> MagicMock:
        fs = MagicMock()
        fs.mime_type.return_value = mime_return
        return fs

    def test_pdf_extension(self):
        fs = self._make_fs()
        mime = QwenVLAdapter._detect_mime_type("invoice.pdf", fs)
        self.assertEqual(mime, QwenVLMimeTypes.PDF)
        fs.mime_type.assert_not_called()

    def test_png_extension(self):
        fs = self._make_fs()
        mime = QwenVLAdapter._detect_mime_type("scan.PNG", fs)
        self.assertEqual(mime, QwenVLMimeTypes.PNG)

    def test_jpeg_extension(self):
        fs = self._make_fs()
        mime = QwenVLAdapter._detect_mime_type("photo.jpeg", fs)
        self.assertEqual(mime, QwenVLMimeTypes.JPEG)

    def test_jpg_extension(self):
        fs = self._make_fs()
        mime = QwenVLAdapter._detect_mime_type("photo.jpg", fs)
        self.assertEqual(mime, QwenVLMimeTypes.JPEG)

    def test_unknown_extension_falls_back_to_sniff(self):
        fs = self._make_fs(mime_return="image/png")
        mime = QwenVLAdapter._detect_mime_type("SOURCE", fs)
        self.assertEqual(mime, "image/png")

    def test_unknown_extension_sniff_fails_returns_pdf(self):
        fs = self._make_fs()
        fs.mime_type.side_effect = Exception("not available")
        mime = QwenVLAdapter._detect_mime_type("SOURCE", fs)
        self.assertEqual(mime, QwenVLMimeTypes.PDF)


# ---------------------------------------------------------------------------
# Base URL / API key resolution
# ---------------------------------------------------------------------------

class TestResolveBaseUrl(unittest.TestCase):

    def test_url_from_config(self):
        adapter = _make_adapter({"base_url": "http://my-server:9000/v1"})
        self.assertEqual(adapter._resolve_base_url(), "http://my-server:9000/v1")

    def test_url_from_env(self):
        adapter = QwenVLAdapter(settings={})
        with patch.dict(os.environ, {"QWEN_BASE_URL": "http://env-server:8000/v1"}):
            self.assertEqual(adapter._resolve_base_url(), "http://env-server:8000/v1")

    def test_missing_url_raises(self):
        adapter = QwenVLAdapter(settings={})
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("QWEN_BASE_URL", None)
            with self.assertRaises(AdapterError):
                adapter._resolve_base_url()

    def test_trailing_slash_stripped(self):
        adapter = _make_adapter({"base_url": "http://localhost:8000/v1/"})
        self.assertEqual(adapter._resolve_base_url(), "http://localhost:8000/v1")


class TestResolveApiKey(unittest.TestCase):

    def test_key_from_config(self):
        adapter = _make_adapter({"api_key": "my-key"})
        self.assertEqual(adapter._resolve_api_key(), "my-key")

    def test_key_from_env(self):
        adapter = QwenVLAdapter(settings={})
        with patch.dict(os.environ, {"QWEN_API_KEY": "env-key"}):
            self.assertEqual(adapter._resolve_api_key(), "env-key")

    def test_missing_key_returns_default(self):
        """API key should default to 'not-needed' for self-hosted servers."""
        adapter = QwenVLAdapter(settings={})
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("QWEN_API_KEY", None)
            self.assertEqual(adapter._resolve_api_key(), QwenVLDefaults.API_KEY)

    def test_empty_config_key_falls_back_to_env(self):
        adapter = QwenVLAdapter(settings={"api_key": ""})
        with patch.dict(os.environ, {"QWEN_API_KEY": "fallback-key"}):
            self.assertEqual(adapter._resolve_api_key(), "fallback-key")


# ---------------------------------------------------------------------------
# process() — happy path with mocked OpenAI
# ---------------------------------------------------------------------------

FAKE_RESPONSE_TEXT = (
    "<<< PAGE_1 >>>\nThis is page one content.\n"
    "<<< PAGE_2 >>>\nThis is page two content."
)


def _patch_openai(response_text: str = FAKE_RESPONSE_TEXT):
    """Return a context-manager that patches openai inside the adapter module."""
    fake_response = _fake_openai_response(response_text)

    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_client.chat.completions.create.return_value = fake_response

    return patch.dict(
        "sys.modules",
        {"openai": mock_openai},
    )


class TestProcess(unittest.TestCase):

    def _make_fs(self, file_bytes: bytes = PDF_MAGIC) -> MagicMock:
        fs = MagicMock()
        fs.read.return_value = file_bytes
        fs.mime_type.return_value = QwenVLMimeTypes.PNG  # simulate image input
        return fs

    def test_basic_extraction_returns_result(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        # Patch to skip PDF-to-image conversion — treat as image input.
        with _patch_openai():
            result = adapter.process(
                input_file_path="invoice.png",
                output_file_path=None,
                fs=fs,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.extracted_text, FAKE_RESPONSE_TEXT)
        self.assertEqual(result.page_count, 2)

    def test_writes_output_file_when_path_given(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with _patch_openai():
            adapter.process(
                input_file_path="invoice.png",
                output_file_path="/tmp/invoice.txt",
                fs=fs,
            )

        fs.write.assert_called()
        first_call = fs.write.call_args_list[0]
        self.assertIn("invoice.txt", str(first_call))

    def test_single_page_no_delimiter(self):
        single_page_text = "No page markers here, just raw text."
        adapter = _make_adapter()
        fs = self._make_fs()

        with _patch_openai(response_text=single_page_text):
            result = adapter.process(input_file_path="doc.png", fs=fs)

        self.assertEqual(result.page_count, 1)
        self.assertIn(single_page_text, result.extracted_text)

    def test_model_name_configurable(self):
        """Confirm that a custom model_name reaches the OpenAI client."""
        adapter = _make_adapter({"model_name": "Qwen/Qwen2.5-VL-7B-Instruct"})
        fs = self._make_fs()

        with _patch_openai() as patches:
            import sys
            mock_openai = sys.modules["openai"]
            mock_client = mock_openai.OpenAI.return_value
            adapter.process(input_file_path="file.png", fs=fs)
            call_args = mock_client.chat.completions.create.call_args
            self.assertIn("Qwen/Qwen2.5-VL-7B-Instruct", str(call_args))

    def test_read_error_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = MagicMock()
        fs.read.side_effect = OSError("Permission denied")
        fs.mime_type.return_value = QwenVLMimeTypes.PNG

        with _patch_openai():
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="invoice.png", fs=fs)
        self.assertIn("Permission denied", str(ctx.exception))

    def test_pdf_input_calls_pdf_conversion(self):
        """Verify that PDF inputs trigger the PDF-to-image conversion."""
        adapter = _make_adapter()
        fs = MagicMock()
        fs.read.return_value = PDF_MAGIC
        fs.mime_type.return_value = QwenVLMimeTypes.PDF

        # Mock the PDF conversion to return a fake image.
        fake_images = [("base64data", "image/png")]

        with _patch_openai():
            with patch.object(
                QwenVLAdapter, "_pdf_to_base64_images", return_value=fake_images
            ) as mock_convert:
                result = adapter.process(input_file_path="doc.pdf", fs=fs)
                mock_convert.assert_called_once()
        self.assertEqual(result.page_count, 2)


# ---------------------------------------------------------------------------
# process() — error cases
# ---------------------------------------------------------------------------

class TestProcessErrors(unittest.TestCase):

    def _make_fs(self, file_bytes: bytes = PDF_MAGIC) -> MagicMock:
        fs = MagicMock()
        fs.read.return_value = file_bytes
        fs.mime_type.return_value = QwenVLMimeTypes.PNG
        return fs

    def _make_failing_openai(self, exc: Exception):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = exc

        # Ensure isinstance checks against openai exception types fail
        # (they're MagicMock, not real exception types).
        mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})

        return patch.dict("sys.modules", {"openai": mock_openai})

    def test_rate_limit_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with self._make_failing_openai(Exception("429 rate limit exceeded")):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.png", fs=fs)
        self.assertIn("rate limit", str(ctx.exception).lower())

    def test_auth_error_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with self._make_failing_openai(Exception("401 Unauthorized")):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.png", fs=fs)
        self.assertIn("auth", str(ctx.exception).lower())

    def test_timeout_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with self._make_failing_openai(Exception("request timeout exceeded")):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.png", fs=fs)
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_connection_refused_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with self._make_failing_openai(Exception("Connection refused")):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.png", fs=fs)
        self.assertIn("connect", str(ctx.exception).lower())

    def test_empty_response_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        empty_message = SimpleNamespace(content="", role="assistant")
        empty_choice = SimpleNamespace(
            message=empty_message, index=0, finish_reason="stop"
        )
        empty_response = SimpleNamespace(
            choices=[empty_choice], usage=None
        )
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = empty_response
        mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})

        with patch.dict("sys.modules", {"openai": mock_openai}):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.png", fs=fs)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_context_size_exceeded_raises_adapter_error(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        with self._make_failing_openai(
            Exception("request (12421 tokens) exceeds the available context size (8192 tokens)")
        ):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.png", fs=fs)
        self.assertIn("context window", str(ctx.exception).lower())

    def test_delimiter_only_response_raises_adapter_error(self):
        """When the model returns only page delimiters with no text,
        it should raise AdapterError with guidance about --mmproj."""
        adapter = _make_adapter()
        fs = self._make_fs()

        # Simulate a response that is just the delimiter — no actual content
        delimiter_only_text = "<<< PAGE_1 >>>"
        with _patch_openai(response_text=delimiter_only_text):
            with self.assertRaises(AdapterError) as ctx:
                adapter.process(input_file_path="file.png", fs=fs)
        error_msg = str(ctx.exception).lower()
        self.assertIn("delimiter", error_msg)
        self.assertIn("--mmproj", str(ctx.exception))

    def test_missing_openai_package_raises(self):
        adapter = _make_adapter()
        fs = self._make_fs()

        import sys
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "openai"}
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'openai'"),
        ):
            with self.assertRaises((AdapterError, ImportError)):
                adapter.process(input_file_path="file.png", fs=fs)
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# test_connection()
# ---------------------------------------------------------------------------

class TestTestConnection(unittest.TestCase):

    def _make_openai_with_response(self):
        fake_response = _fake_openai_response("Hi there")
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = fake_response
        return mock_openai

    def test_success_returns_true(self):
        adapter = _make_adapter()
        mock_openai = self._make_openai_with_response()

        with patch.dict("sys.modules", {"openai": mock_openai}):
            self.assertTrue(adapter.test_connection())

    def test_connection_failure_raises_adapter_error(self):
        adapter = _make_adapter()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception(
            "Connection refused"
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            with self.assertRaises(AdapterError):
                adapter.test_connection()

    def test_empty_choices_raises(self):
        adapter = _make_adapter()
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[], usage=None
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            with self.assertRaises(AdapterError):
                adapter.test_connection()

    def test_missing_base_url_raises(self):
        adapter = QwenVLAdapter(settings={})
        mock_openai = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("QWEN_BASE_URL", None)
            with patch.dict("sys.modules", {"openai": mock_openai}):
                with self.assertRaises(AdapterError):
                    adapter.test_connection()


# ---------------------------------------------------------------------------
# Registry metadata check
# ---------------------------------------------------------------------------

class TestRegistryMetadata(unittest.TestCase):

    def test_metadata_has_required_keys(self):
        from unstract.sdk1.adapters.x2text.qwen_vl.src import metadata

        for key in ("name", "version", "adapter", "description", "is_active"):
            self.assertIn(key, metadata, f"Missing metadata key: {key}")

    def test_is_active(self):
        from unstract.sdk1.adapters.x2text.qwen_vl.src import metadata

        self.assertTrue(metadata["is_active"])

    def test_adapter_id_unique_format(self):
        adapter_id = QwenVLAdapter.get_id()
        self.assertIn("|", adapter_id, "ID should follow '{name}|{uuid}' format")
        self.assertTrue(adapter_id.startswith("qwenvl|"))

    def test_adapter_type_is_x2text(self):
        from unstract.sdk1.adapters.enums import AdapterTypes

        self.assertEqual(QwenVLAdapter.get_adapter_type(), AdapterTypes.X2TEXT)


# ---------------------------------------------------------------------------
# Custom prompt building
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt(unittest.TestCase):

    def test_base_prompt_without_custom(self):
        adapter = _make_adapter()
        prompt = adapter._build_system_prompt()
        self.assertIn("layout-preserving OCR", prompt)
        self.assertNotIn("ADDITIONAL INSTRUCTIONS", prompt)

    def test_custom_prompt_appended(self):
        adapter = _make_adapter({"custom_prompt": "Focus on tables only."})
        prompt = adapter._build_system_prompt()
        self.assertIn("ADDITIONAL INSTRUCTIONS FROM USER", prompt)
        self.assertIn("Focus on tables only.", prompt)

    def test_empty_custom_prompt_ignored(self):
        adapter = _make_adapter({"custom_prompt": "   "})
        prompt = adapter._build_system_prompt()
        self.assertNotIn("ADDITIONAL INSTRUCTIONS", prompt)


if __name__ == "__main__":
    unittest.main()


# Unstract Gemini 2.0 Flash X2Text Adapter

A native Text Extractor adapter for Unstract that uses Google's `gemini-2.0-flash`
multimodal model as a layout-preserving document parser and OCR engine.

## Overview

This adapter sends document bytes (PDF or image formats) directly to Gemini 2.0 Flash
as an inline multimodal part and uses a structured prompt to coax layout-preserving OCR
output — preserving tables, columns, form key-value pairs, and page boundaries — that
matches Unstract's standard `TextExtractionResult` contract.

## Configuration

| Field           | Type       | Required | Default              | Description                                     |
|-----------------|------------|----------|----------------------|-------------------------------------------------|
| `adapter_name`  | `string`   | Yes      | `gemini-flash-1`     | Unique name for this adapter instance.          |
| `api_key`       | `password` | No       | –                    | Google AI Studio API Key. Falls back to the `GEMINI_API_KEY` env variable. |
| `model_name`    | `string`   | No       | `gemini-2.0-flash`   | Gemini model identifier to use.                 |
| `custom_prompt` | `textarea` | No       | –                    | Additional instructions appended to the system prompt. |

## Env Variables

| Variable          | Description                                              |
|-------------------|----------------------------------------------------------|
| `GEMINI_API_KEY`  | Fallback API key when none is set in the adapter config. |

## Supported Input Formats

- PDF (`application/pdf`)
- PNG (`image/png`)
- JPEG / JPG (`image/jpeg`)
- GIF (`image/gif`)
- WebP (`image/webp`)

## Extraction Behaviour

- Pages are delimited in Gemini's output with `<<< PAGE_{n} >>>` markers.
- Tables are rendered as Markdown tables.
- Column boundaries and form key-value pairs are preserved.
- The returned `TextExtractionResult.extracted_text` is the full concatenated text.
- Token usage and model name are recorded in a sidecar metadata JSON file (same
  pattern as LLMWhisperer).

## Error Handling

| Scenario             | Raised exception  |
|----------------------|-------------------|
| API key missing      | `AdapterError`    |
| Rate limit (HTTP 429)| `AdapterError`    |
| Timeout / network    | `AdapterError`    |
| Unsupported MIME     | `AdapterError`    |
| Empty API response   | `AdapterError`    |


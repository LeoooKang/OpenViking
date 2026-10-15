# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for OpenAI request defaults and explicitly configured reasoning effort."""

from types import SimpleNamespace

import pytest

from openviking.models.vlm.backends.openai_vlm import OpenAIVLM
from openviking_cli.utils.config.vlm_config import VLMConfig


class TestOpenAITextCompletionParams:
    """OpenAI defaults must not gate compatible-provider configuration."""

    @pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-6-luna", "GPT-7"])
    def test_reasoning_families_use_max_completion_tokens(self, model):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": model,
                "api_base": "https://api.openai.com/v1",
                "max_tokens": 512,
            }
        )
        kwargs = vlm._build_text_kwargs(prompt="hi", max_tokens=128)
        assert kwargs["max_completion_tokens"] == 128
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs
        assert kwargs["reasoning_effort"] == "low"

    def test_o3_mini_uses_max_completion_tokens(self):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "o3-mini",
                "api_base": "https://api.openai.com/v1",
                "max_tokens": 256,
            }
        )
        kwargs = vlm._build_text_kwargs(prompt="hi")
        assert kwargs["max_completion_tokens"] == 256
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs
        assert kwargs["reasoning_effort"] == "low"

    def test_reasoning_effort_overridable_via_config(self):
        vlm = VLMConfig(
            provider="glm",
            model="glm-5.3-flash",
            api_key="sk-test",
            max_tokens=512,
            temperature=0.3,
            reasoning_effort="high",
        ).get_vlm_instance()
        kwargs = vlm._build_text_kwargs(prompt="hi", max_tokens=128)
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_tokens"] == 128
        assert "max_completion_tokens" not in kwargs
        assert kwargs["temperature"] == 0.3

    def test_gpt4o_mini_keeps_max_tokens_and_temperature(self):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "gpt-4o-mini",
                "api_base": "https://api.openai.com/v1",
                "max_tokens": 512,
                "temperature": 0.5,
            }
        )
        kwargs = vlm._build_text_kwargs(prompt="hi")
        assert kwargs["max_tokens"] == 512
        assert "max_completion_tokens" not in kwargs
        assert kwargs["temperature"] == 0.5
        assert "reasoning_effort" not in kwargs

    def test_reasoning_model_without_max_tokens_omits_both(self):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "gpt-5",
                "api_base": "https://api.openai.com/v1",
            }
        )
        kwargs = vlm._build_text_kwargs(prompt="hi")
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs
        assert "temperature" not in kwargs


class TestOpenAIVisionCompletionParams:
    """Vision requests preserve the same OpenAI defaults as text requests."""

    def test_gpt5_mini_vision_uses_max_completion_tokens(self):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "gpt-5-mini",
                "api_base": "https://api.openai.com/v1",
                "max_tokens": 1024,
            }
        )
        kwargs = vlm._build_vision_kwargs(prompt="describe this")
        assert kwargs["max_completion_tokens"] == 1024
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs
        assert kwargs["reasoning_effort"] == "low"

    def test_gpt4o_vision_keeps_max_tokens_and_temperature(self):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "gpt-4o",
                "api_base": "https://api.openai.com/v1",
                "max_tokens": 1024,
                "temperature": 0.2,
            }
        )
        kwargs = vlm._build_vision_kwargs(prompt="describe this")
        assert kwargs["max_tokens"] == 1024
        assert "max_completion_tokens" not in kwargs
        assert kwargs["temperature"] == 0.2


class TestOpenAIOutputEffortParams:
    """output_effort configuration for MiniMax and compatible models."""

    def test_output_effort_via_vlm_config(self):
        vlm = VLMConfig(
            provider="openai",
            model="MiniMax-M3.1-Flash-Preview",
            api_key="sk-test",
            api_base="https://api.minimaxi.com/v1",
            output_effort="max",
        ).get_vlm_instance()
        kwargs = vlm._build_text_kwargs(prompt="hello")
        assert kwargs["extra_body"] == {"output_config": {"effort": "max"}}

    def test_output_effort_preserves_existing_extra_request_body(self):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "MiniMax-M3.1-Flash-Preview",
                "api_base": "https://api.minimaxi.com/v1",
                "extra_request_body": {"custom_flag": True},
                "output_effort": "high",
            }
        )
        kwargs = vlm._build_text_kwargs(prompt="hello")
        assert kwargs["extra_body"] == {
            "custom_flag": True,
            "output_config": {"effort": "high"},
        }

    def test_output_effort_per_call_override(self):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "MiniMax-M3.1-Flash-Preview",
                "api_base": "https://api.minimaxi.com/v1",
                "output_effort": "low",
            }
        )
        kwargs = vlm._build_text_kwargs(prompt="hello", output_effort="max")
        assert kwargs["extra_body"] == {"output_config": {"effort": "max"}}


class TestOpenAIExtractionTimeoutParams:
    """extraction_timeout_s configuration and propagation for extraction calls."""

    def test_extraction_timeout_via_vlm_config(self):
        config = VLMConfig(
            provider="openai",
            model="MiniMax-M3.1-Flash-Preview",
            api_key="sk-test",
            api_base="https://api.minimaxi.com/v1",
            timeout=600.0,
            extraction_timeout_s=3000.0,
        )
        assert config.extraction_timeout_s == 3000.0
        config_dict = config._build_vlm_config_dict()
        assert config_dict["timeout"] == 600.0
        assert config_dict["extraction_timeout_s"] == 3000.0
        vlm = config.get_vlm_instance()
        assert getattr(vlm, "extraction_timeout_s", None) == 3000.0

    def test_extraction_timeout_via_credential(self):
        config = VLMConfig(
            provider="openai",
            model="MiniMax-M3.1-Flash-Preview",
            credentials=[
                {
                    "provider": "openai",
                    "model": "MiniMax-M3.1-Flash-Preview",
                    "api_key": "sk-cred-1",
                    "extraction_timeout_s": 2500.0,
                }
            ],
            timeout=600.0,
            extraction_timeout_s=3000.0,
        )
        cred = config.credentials[0]
        assert cred.extraction_timeout_s == 2500.0
        config_dict = config._build_vlm_config_dict_for_credential(cred)
        assert config_dict["extraction_timeout_s"] == 2500.0

    def test_extraction_timeout_credential_inherits_parent(self):
        config = VLMConfig(
            provider="openai",
            model="MiniMax-M3.1-Flash-Preview",
            credentials=[
                {
                    "provider": "openai",
                    "model": "MiniMax-M3.1-Flash-Preview",
                    "api_key": "sk-cred-1",
                }
            ],
            timeout=600.0,
            extraction_timeout_s=3000.0,
        )
        cred = config.credentials[0]
        assert cred.extraction_timeout_s is None
        config_dict = config._build_vlm_config_dict_for_credential(cred)
        assert config_dict["extraction_timeout_s"] == 3000.0

    def test_openai_vlm_accepts_extraction_timeout(self):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "MiniMax-M3.1-Flash-Preview",
                "timeout": 600.0,
                "extraction_timeout_s": 3000.0,
            }
        )
        assert vlm.timeout == 600.0
        assert vlm.extraction_timeout_s == 3000.0

    def test_openai_vlm_get_completion_forwards_timeout(self, monkeypatch):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "MiniMax-M3.1-Flash-Preview",
                "timeout": 600.0,
                "extraction_timeout_s": 3000.0,
            }
        )
        captured_kwargs = {}

        class DummyCompletions:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                response = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="test response"))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                )
                return response

        class DummyClient:
            chat = SimpleNamespace(completions=DummyCompletions())

        monkeypatch.setattr(vlm, "get_client", lambda: DummyClient())

        # When timeout is explicitly passed, it overrides the client call
        res = vlm.get_completion(prompt="hello", timeout=3000.0)
        assert res == "test response"
        assert captured_kwargs.get("timeout") == 3000.0

        # When timeout is omitted, timeout is not injected into call_kwargs
        captured_kwargs.clear()
        res = vlm.get_completion(prompt="hello")
        assert res == "test response"
        assert "timeout" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_openai_vlm_get_completion_async_forwards_timeout(self, monkeypatch):
        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "model": "MiniMax-M3.1-Flash-Preview",
                "timeout": 600.0,
                "extraction_timeout_s": 3000.0,
            }
        )
        captured_kwargs = {}

        class DummyAsyncCompletions:
            async def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                response = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="async response"))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                )
                return response

        class DummyAsyncClient:
            chat = SimpleNamespace(completions=DummyAsyncCompletions())

        monkeypatch.setattr(vlm, "get_async_client", lambda: DummyAsyncClient())

        res = await vlm.get_completion_async(prompt="hello", timeout=3000.0)
        assert res == "async response"
        assert captured_kwargs.get("timeout") == 3000.0

        captured_kwargs.clear()
        res = await vlm.get_completion_async(prompt="hello")
        assert res == "async response"
        assert "timeout" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_extract_loop_resolves_and_passes_timeout(self):
        from unittest.mock import AsyncMock, MagicMock
        from openviking.session.memory.extract_loop import ExtractLoop

        mock_vlm = MagicMock()
        mock_vlm.model = "test-model"
        mock_vlm.extraction_timeout_s = 3000.0
        mock_vlm.get_completion_async = AsyncMock(return_value="done")

        loop = ExtractLoop(
            vlm=mock_vlm,
            viking_fs=MagicMock(),
            context_provider=MagicMock(),
        )
        assert loop._effective_timeout == 3000.0

        await loop._call_llm(messages=[{"role": "user", "content": "hi"}])
        assert mock_vlm.get_completion_async.call_count == 1
        _, kwargs = mock_vlm.get_completion_async.call_args
        assert kwargs["timeout"] == 3000.0




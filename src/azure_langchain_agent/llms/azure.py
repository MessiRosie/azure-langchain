"""Azure OpenAI legacy LLM (text completions) — minimal implementation."""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from openai import AsyncAzureOpenAI, AzureOpenAI as AzureOpenAIClient


class AzureOpenAI(BaseLLM):
    """A minimal Azure OpenAI legacy text-completion model.

    Args:
        azure_deployment: name of the deployment in Azure OpenAI.
        azure_endpoint: your Azure OpenAI resource endpoint URL.
        api_key: API key. If omitted, ``AZURE_OPENAI_API_KEY`` is read from env.
        api_version: Azure OpenAI API version.
        temperature: sampling temperature.
        max_tokens: maximum number of tokens to generate.
    """

    azure_deployment: str
    azure_endpoint: str
    api_key: Optional[str] = None
    api_version: str = "2024-05-01-preview"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "azure-openai"

    def _client(self) -> AzureOpenAIClient:
        return AzureOpenAIClient(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def _aclient(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        client = self._client()
        generations = []
        for prompt in prompts:
            response = client.completions.create(
                model=self.azure_deployment,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
            generations.append([Generation(text=response.choices[0].text or "")])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        client = self._aclient()
        generations = []
        for prompt in prompts:
            response = await client.completions.create(
                model=self.azure_deployment,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
            generations.append([Generation(text=response.choices[0].text or "")])
        return LLMResult(generations=generations)

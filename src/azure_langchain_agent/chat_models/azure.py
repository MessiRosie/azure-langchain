"""Azure OpenAI chat model — minimal langchain-compatible implementation."""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import AsyncAzureOpenAI, AzureOpenAI

_ROLE_MAP = {
    HumanMessage: "user",
    AIMessage: "assistant",
    SystemMessage: "system",
}


def _to_openai_message(message: BaseMessage) -> dict:
    role = _ROLE_MAP.get(type(message), "user")
    return {"role": role, "content": message.content}


class AzureChatOpenAI(BaseChatModel):
    """A minimal Azure OpenAI chat model.

    Args:
        azure_deployment: name of the model deployment in Azure OpenAI.
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
        return "azure-chat-openai"

    def _client(self) -> AzureOpenAI:
        return AzureOpenAI(
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self._client().chat.completions.create(
            model=self.azure_deployment,
            messages=[_to_openai_message(m) for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
        )
        content = response.choices[0].message.content or ""
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))],
            llm_output={"model": response.model},
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = await self._aclient().chat.completions.create(
            model=self.azure_deployment,
            messages=[_to_openai_message(m) for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
        )
        content = response.choices[0].message.content or ""
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))],
            llm_output={"model": response.model},
        )

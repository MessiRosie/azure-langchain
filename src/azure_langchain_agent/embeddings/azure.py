"""Azure OpenAI embeddings — minimal implementation."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.embeddings import Embeddings
from openai import AsyncAzureOpenAI, AzureOpenAI


class AzureOpenAIEmbeddings(Embeddings):
    """A minimal Azure OpenAI embeddings model.

    Args:
        azure_deployment: name of the embedding deployment in Azure OpenAI.
        azure_endpoint: your Azure OpenAI resource endpoint URL.
        api_key: API key. If omitted, ``AZURE_OPENAI_API_KEY`` is read from env.
        api_version: Azure OpenAI API version.
    """

    def __init__(
        self,
        azure_deployment: str,
        azure_endpoint: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-05-01-preview",
    ) -> None:
        self.azure_deployment = azure_deployment
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version

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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._client().embeddings.create(
            model=self.azure_deployment, input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self._aclient().embeddings.create(
            model=self.azure_deployment, input=texts
        )
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]

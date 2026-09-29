"""azure_langchain_agent — a minimal LangChain wrapper for Azure OpenAI.

Provides ``AzureChatOpenAI`` (chat), ``AzureOpenAI`` (legacy completions) and
``AzureOpenAIEmbeddings`` (embeddings), implemented on top of the official
``openai`` SDK and the ``langchain-core`` interfaces.
"""

__version__ = "0.1.0"

# --- Exports ---
from .chat_models.azure import AzureChatOpenAI
from .embeddings.azure import AzureOpenAIEmbeddings
from .llms.azure import AzureOpenAI

__all__ = ["AzureChatOpenAI", "AzureOpenAI", "AzureOpenAIEmbeddings"]

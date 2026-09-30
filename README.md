# Azure LangChain Agent

一个用于 Azure OpenAI 与 LangChain 集成的最小 Python 项目示例。自实现 `AzureChatOpenAI`、
`AzureOpenAI` 和 `AzureOpenAIEmbeddings` 三个封装，基于官方 `openai` SDK 与 `langchain-core` 接口。

## 安装

```bash
pip install git+https://github.com/MessiRosie/langchain-openai.git
```

或在项目根目录本地安装：

```bash
pip install -e .
```

## 配置

先在 Azure OpenAI 中创建模型部署，然后设置环境变量：

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
```

## 使用 Azure OpenAI 聊天模型

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="your-deployment-name",
    azure_endpoint="https://your-resource-name.openai.azure.com/",
    api_version="2024-05-01-preview",
    temperature=0,
)

response = llm.invoke("用一句话解释什么是 RAG。")
print(response.content)
```

`azure_deployment` 是 Azure 门户中创建的部署名称，可能与底层模型名称不同。

## 使用向量嵌入

```python
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="your-embedding-deployment",
    azure_endpoint="https://your-resource-name.openai.azure.com/",
)

vectors = embeddings.embed_documents(["hello", "world"])
query_vec = embeddings.embed_query("hello")
```

## 使用文本补全（Legacy）

```python
from langchain_openai import AzureOpenAI

llm = AzureOpenAI(
    azure_deployment="your-deployment-name",
    azure_endpoint="https://your-resource-name.openai.azure.com/",
)

print(llm.invoke("Say hi"))
```

## 更新

```bash
git add . && git commit -m "..." && git push
```

## License

MIT

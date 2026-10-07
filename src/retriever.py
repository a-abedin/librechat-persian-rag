import os
from typing import List
try:
  from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
  try:
    from langchain_classic.retrievers import EnsembleRetriever
  except (ImportError, ModuleNotFoundError):
    from langchain_community.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import AppConfig


class HybridRetrieverManager:
    """مدیریت امبدینگ‌ها، پایگاه برداری محلی و رتریور ترکیبی (Dense + Sparse)"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "۔ ",
                "؟ ",
                "! ",
                "، ",
                "؛ ",
                " ",
                "",
            ],
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_or_load_retriever(
        self, raw_documents: List[Document]
    ) -> BaseRetriever:
        splitted_docs = self.splitter.split_documents(raw_documents)

        # ساخت یا بازخوانی چروما
        if os.path.exists(self.config.persist_directory) and len(
            os.listdir(self.config.persist_directory)
        ) > 0:
            vectorstore = Chroma(
                persist_directory=self.config.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.config.collection_name,
            )
        else:
            vectorstore = Chroma.from_documents(
                documents=splitted_docs,
                embedding=self.embedding_model,
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
            )

        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # ساخت BM25 برای تطابق نام‌های خاص
        bm25_retriever = BM25Retriever.from_documents(splitted_docs)
        bm25_retriever.k = 5

        # رتریور هیبریدی
        return EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.3, 0.7],
        )

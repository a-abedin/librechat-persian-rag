import os
import random
import time
from typing import List
from langchain_community.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
    PyPDFium2Loader,
    WebBaseLoader,
    WikipediaLoader,
)
from langchain_core.documents import Document

from src.preprocessor import PersianTextCleaner


class MultiSourceDataLoader:
    """بارگیری و یکپارچه‌سازی اسناد از PDF، وب، ویکی‌پدیا و آرشیوهای محلی"""

    def __init__(self, cleaner: PersianTextCleaner):
        self.cleaner = cleaner

    def load_pdf(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            return []
        loader = PyPDFium2Loader(file_path)
        docs = loader.load()
        return self._clean_and_filter(docs)

    def load_web_page(self, url: str) -> List[Document]:
        loader = WebBaseLoader(web_path=url)
        docs = loader.load()
        return self._clean_and_filter(docs)

    def load_wikipedia_topics(
        self, titles: List[str], max_retries: int = 3
    ) -> List[Document]:
        wiki_docs = []
        for title in titles:
            for attempt in range(max_retries):
                try:
                    loader = WikipediaLoader(
                        query=title, lang="fa", load_max_docs=1
                    )
                    docs = loader.load()
                    if docs:
                        wiki_docs.extend(docs)
                    break
                except Exception:
                    time.sleep(random.uniform(2.0, 4.0) * (attempt + 1))
            time.sleep(random.uniform(1.0, 2.5))
        return self._clean_and_filter(wiki_docs)

    def load_html_directory(self, directory_path: str) -> List[Document]:
        if not os.path.isdir(directory_path):
            return []
        loader = DirectoryLoader(
            path=directory_path,
            glob="**/*.html",
            loader_cls=BSHTMLLoader,
            loader_kwargs={
                "open_encoding": "utf-8",
                "bs_kwargs": {"features": "html.parser"},
            },
        )
        docs = loader.load()
        return self._clean_and_filter(docs)

    def _clean_and_filter(self, docs: List[Document]) -> List[Document]:
        cleaned = []
        for doc in docs:
            content = self.cleaner.clean(doc.page_content)
            if len(content.strip()) > 20:
                doc.page_content = content
                cleaned.append(doc)
        return cleaned

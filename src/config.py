import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    # مسیرهای داده
    pdf_path: str = "data/justforfun_persian.pdf"
    html_dir: str = "data/html/"
    web_url: str = "https://linuxbook.ir/all.html"
    wiki_titles: List[str] = field(
        default_factory=lambda: [
            "ریچارد استالمن",
            "لینوس توروالدز",
            "لینوکس",
            "پروژه گنو",
            "نرم‌افزار آزاد",
            "بنیاد نرم‌افزار آزاد",
        ]
    )

    # تنظیمات پایگاه برداری و چانکینگ
    persist_directory: str = "./chroma_librechat_db"
    collection_name: str = "librechat_knowledge_base"
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # مدل زبانی
    llm_model_name: str = "command-r-plus-08-2024"
    temperature: float = 0.0
    max_answer_words: int = 4


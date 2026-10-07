import re
import time
from typing import Any, Dict
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough

from src.config import AppConfig


class LibreChatEngine:
    """موتور RAG برای پاسخ‌گویی به پرسش‌های کوتاه و مقید به کانتکست"""

    PROMPT_TEMPLATE = """شما یک دستیار استخراج اطلاعات فوق‌العاده دقیق درباره لینوکس، نرم‌افزار آزاد، استالمن و توروالدز هستید.
با توجه به متون زمینه زیر به سوال کاربر پاسخ دهید.

متون زمینه:
{context}

دستورالعمل‌های حیاتی:
۱. پاسخ را حتماً به زبان فارسی و حداکثر در ۱ تا ۴ کلمه بنویسید (بدون مقدمه و بدون جمله‌بندی).
۲. اگر سوال نام یک کشور یا مکان را می‌خواهد، تنها نام فارسی آن کشور را بنویسید.
۳. نام‌ها و اصطلاحات خاص را به دقیق‌ترین شکل ممکن از متن استخراج کنید.
۴. اگر پاسخ در متن وجود نداشت، بنویسید: نمی‌دانم

سوال: {question}
پاسخ:"""

    def __init__(self, retriever: BaseRetriever, config: AppConfig):
        self.config = config
        self.retriever = retriever
        self.model = ChatCohere(
            model=config.llm_model_name,
            temperature=config.temperature,
        )
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt = PromptTemplate.from_template(self.PROMPT_TEMPLATE)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            RunnablePassthrough.assign(
                context=(lambda x: x["question"])
                | self.retriever
                | format_docs
            )
            | prompt
            | self.model
            | StrOutputParser()
        )

    def answer_question(self, query: str, question_number: int) -> Dict[str, Any]:
        clean_q = re.sub(
            r"^پرسش\s*[\d\u06F0-\u06F9\u0660-\u0669]+[:\s\-]+", "", query
        ).strip()

        try:
            raw_answer = self.chain.invoke({"question": clean_q})
            processed_answer = self._apply_guardrails(raw_answer)
        except Exception as e:
            print(f"[Engine Error] Question {question_number}: {e}")
            processed_answer = "نمی‌دانم"

        # حفظ سهمیه درخواست API
        time.sleep(2.0)

        return {
            "question_number": question_number,
            "answer": processed_answer,
        }

    def _apply_guardrails(self, text: str) -> str:
        cleaned = str(text).strip().replace("\n", " ")
        cleaned = re.sub(r'[\.،؛!؟"\'`]+$', "", cleaned).strip()

        # رفع باگ املایی ناشی از PDF
        if "سکسیمبولی" in cleaned:
            cleaned = "سیمبولیکس"

        words = cleaned.split()
        if 0 < len(words) <= self.config.max_answer_words:
            return cleaned
        elif len(words) > self.config.max_answer_words:
            return " ".join(words[: self.config.max_answer_words])
        return "نمی‌دانم"

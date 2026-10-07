import argparse
import json
from src.config import AppConfig
from src.loaders import MultiSourceDataLoader
from src.preprocessor import PersianTextCleaner
from src.rag_engine import LibreChatEngine
from src.retriever import HybridRetrieverManager

BENCHMARK_QUESTIONS = [
    "پرسش ۱: توروالدز برای کار در چه موسسه‌ای دانشگاه هلسینکی را ترک گفت؟",
    "پرسش ۲: آندرو تاننباوم استاد کدام دانشگاه است؟",
    (
        "پرسش ۳: در سال ۲۰۰۶ چند درصد از هسته لینوکس توسط توروالدز نوشته شد"
        " (به عدد)؟"
    ),
    "پرسش ۴: چه کسی بنیاد نرم‌افزارهای آزاد را بنا نهاد؟",
    "پرسش ۵: ریچارد استالمن در ۲۱ سالگی در کدام شرکت کار می‌کرد؟",
    (
        "پرسش ۶: یکی از مشهورترین پروژه‌هایی که در ابتدا پروژه‌ی آزاد و آکادمیک"
        " بود اما بعد وارد محیط بسته‌ی تجاری شد چه بود؟"
    ),
    "پرسش ۷: لینکدین در سانسور کردن حساب‌ها به درخواست چه کشوری مشهور است؟",
    (
        "پرسش ۸: ریچارد استالمن پیشنهاد می‌کند به‌جای گوگل مپ از چه سرویسی"
        " استفاده کنیم؟"
    ),
    "پرسش ۹: آزادی صفرم در نرم‌افزار آزاد چه عنوانی دارد؟",
    "پرسش ۱۰: آیا یک نرم‌افزار آزاد لزوماً رایگان است (بله یا خیر)؟",
    (
        "پرسش ۱۱: استاندارد ناظر بر فایل‌ها و دایرکتوری‌ها به‌اختصار چه نامیده"
        " می‌شود؟"
    ),
    "پرسش ۱۲: اولین ریپلای به ایمیل درخواست کار چیست؟",
    (
        "پرسش ۱۳: اگر امروز که از شنبه ورزش می‌کنم در واقع دچار چه بایاسی"
        " شده‌ایم؟"
    ),
    (
        "پرسش ۱۴: دنبال یاد گرفتن کدوم یکی باشیم: برنامه‌نویسی یا دستور زبان"
        " یک زبان خاص؟"
    ),
    (
        "پرسش ۱۵: اگه هدف‌مون اینه که بریم گوگل کار کنیم اول از همه چه‌چیزی رو"
        " سرچ کنیم؟"
    ),
    (
        "پرسش ۱۶: در بیانیه‌ی هکرها گفته شده که جرم آن‌ها در یک کلمه چیست؟"
    ),
]


def run_pipeline():
    config = AppConfig()
    cleaner = PersianTextCleaner()
    loader = MultiSourceDataLoader(cleaner)

    print("1. Loading raw documents...")
    docs = []
    docs.extend(loader.load_pdf(config.pdf_path))
    docs.extend(loader.load_web_page(config.web_url))
    docs.extend(loader.load_wikipedia_topics(config.wiki_titles))
    docs.extend(loader.load_html_directory(config.html_dir))
    print(f"Total documents loaded: {len(docs)}")

    print("2. Initializing Hybrid Vector & Keyword Retrieval...")
    retriever_mgr = HybridRetrieverManager(config)
    retriever = retriever_mgr.build_or_load_retriever(docs)

    print("3. Building RAG Engine...")
    engine = LibreChatEngine(retriever, config)

    print("4. Running Benchmark Evaluation...")
    answers = []
    for idx, query in enumerate(BENCHMARK_QUESTIONS, start=1):
        print(f"Processing Q{idx}...")
        ans_dict = engine.answer_question(query, idx)
        answers.append(ans_dict)
        print(f"-> {ans_dict}")

    output_file = "answers.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    print(f"\nAll answers exported to {output_file} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibreChat RAG Pipeline")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full 16-question evaluation",
    )
    args = parser.parse_args()
    run_pipeline()

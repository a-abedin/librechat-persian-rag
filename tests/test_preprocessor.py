import pytest
from src.preprocessor import PersianTextCleaner

@pytest.fixture
def cleaner():
    return PersianTextCleaner(fix_word_order=False)

def test_empty_input(cleaner):
    assert cleaner.clean("") == ""
    assert cleaner.clean(None) == ""

def test_arabic_to_persian_standardization(cleaner):
    # تبدیل ي و ك عربی به فارسی
    raw_text = "كتاب يوسف و على"
    expected = "کتاب یوسف و علی"
    assert cleaner.clean(raw_text) == expected

def test_diacritics_and_tashkeel_removal(cleaner):
    # حذف اعراب
    raw_text = "سَلَامٌ عَلَيْكُمْ"
    assert cleaner.clean(raw_text) == "سلام علیکم"

def test_corrupted_cmap_greek_characters(cleaner):
    # اصلاح نگاشت کاراکترهای یونانی ناشی از فونت خراب PDF
    # ͷ -> ک  |  ͽ -> گ  |  ͬ -> ی
    raw_text = "ͷتاب و دانـͽاه"
    cleaned = cleaner.clean(raw_text)
    assert "ک" in cleaned
    assert "گ" in cleaned

def test_specific_word_repair(cleaner):
    # اصلاح واژه‌های معکوس‌شده خاص
    assert cleaner.clean("کشری") == "شریک"
    assert cleaner.clean("ککم") == "کمک"

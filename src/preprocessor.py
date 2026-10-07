import re
from typing import Dict


class PersianTextCleaner:
    """مسئولیت تک‌منظوره (SRP): پالایش، استانداردسازی یونیکد و رفع باگ‌های نویسه‌ای اسناد فارسی"""

    # نگاشت کاراکترهای خراب فونت CMap در PDF
    GREEK_CHAR_MAP: Dict[str, str] = {
        "ͷ": "ک",
        "ͺ": "ک",
        "ͽ": "گ",
        "ͬ": "ی",
    }

    SPECIFIC_REPLACEMENTS: Dict[str, str] = {
        r"\bیم\b": "می",
        r"\bینم\b": "نمی",
        r"\bکی\b": "یک",
        r"\bیول\b": "ولی",
        r"\bکشری\b": "شریک",
        r"\bککم\b": "کمک",
        r"\bکگرافی\b": "گرافیک",
        r"\bیدرک\b": "درکی",
        r"\bیدوم\b": "دومی",
        r"\bنیدکب\b": "بکنید",
        r"\bندکب\b": "بکنید",
        r"\bریگدی\b": "دیگری",
    }

    def __init__(self, fix_word_order: bool = True):
        self.fix_word_order = fix_word_order
        self._tashkeel_pattern = re.compile(r"[\u064B-\u0652\u0670]")

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self._normalize_arabic_characters(text)
        text = self._remove_diacritics_and_special_spaces(text)
        text = self._fix_corrupted_font_artifacts(text)

        if self.fix_word_order:
            text = self._reorder_visual_lines(text)

        text = self._fix_persian_affixes(text)
        return text.strip()

    def _normalize_arabic_characters(self, text: str) -> str:
        char_map = {
            "ي": "ی",
            "ك": "ک",
            "ى": "ی",
            "ة": "ه",
            "ؤ": "و",
            "إ": "ا",
            "أ": "ا",
            "ئ": "ی",
        }
        for ar, fa in char_map.items():
            text = text.replace(ar, fa)
        return text

    def _remove_diacritics_and_special_spaces(self, text: str) -> str:
        text = self._tashkeel_pattern.sub("", text)
        text = re.sub(r"[\u200b\u200d\ufeff]", "", text)
        text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f]", " ", text)
        text = re.sub(r"\s*\u200c\s*", "\u200c", text)
        return text

    def _fix_corrupted_font_artifacts(self, text: str) -> str:
        text = re.sub(r"([آ-ی]+)ͺ([آ-ی]+)", r"\2ک\1", text)
        text = re.sub(r"([آ-ی]+)ͽ([آ-ی]+)", r"\2گ\1", text)
        text = re.sub(r"\bͬ([آ-ی]+)\b", r"\1ی", text)
        text = re.sub(r"\bͷ([آ-ی]+)\b", r"\1ک", text)

        for corrupted, correct in self.GREEK_CHAR_MAP.items():
            text = text.replace(corrupted, correct)

        for pattern, replacement in self.SPECIFIC_REPLACEMENTS.items():
            text = re.sub(pattern, replacement, text)
        return text

    def _reorder_visual_lines(self, text: str) -> str:
        lines = text.split("\n")
        corrected_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                corrected_lines.append("")
                continue

            if re.search(r"^\d+\s+[A-Z]+", line_str) or line_str.isascii():
                corrected_lines.append(line_str)
            else:
                tokens = line_str.split()
                reversed_line = " ".join(tokens[::-1])
                reversed_line = re.sub(
                    r"(^|\s)([\.،!؟؛:])([آ-یa-zA-Z0-9]+)", r"\1\3\2", reversed_line
                )
                reversed_line = re.sub(r"\s+([،؛])\s*", r"\1 ", reversed_line)
                corrected_lines.append(reversed_line)
        return "\n".join(corrected_lines)

    def _fix_persian_affixes(self, text: str) -> str:
        text = re.sub(r"\b(ن?می)\s+", r"\1‌", text)
        text = re.sub(
            r"\s+(ها|هایش|هایی|های|تان|تر|ترین)(?=[\s\.،!؟؛:]|$)", r"‌\1", text
        )
        text = text.replace("١", "۱").replace("٢", "۲").replace("٣", "۳")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text

"""
tests/test_i18n.py - Ensures bilingual parity between Arabic and English translations.
"""
import json
from pathlib import Path

import config


def test_i18n_parity():
    ar_path = config.BASE_DIR / "web" / "i18n" / "ar.json"
    en_path = config.BASE_DIR / "web" / "i18n" / "en.json"

    assert ar_path.exists(), "web/i18n/ar.json must exist"
    assert en_path.exists(), "web/i18n/en.json must exist"

    with open(ar_path, "r", encoding="utf-8") as f:
        ar_data = json.load(f)

    with open(en_path, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    ar_keys = set(ar_data.keys())
    en_keys = set(en_data.keys())

    missing_in_en = ar_keys - en_keys
    missing_in_ar = en_keys - ar_keys

    assert len(missing_in_en) == 0, f"Keys in Arabic but missing in English: {missing_in_en}"
    assert len(missing_in_ar) == 0, f"Keys in English but missing in Arabic: {missing_in_ar}"
    assert len(ar_keys) > 20, "Translation files must contain substantial UI strings"

    # Ensure no values are empty strings
    for k, v in ar_data.items():
        assert isinstance(v, str) and len(v.strip()) > 0, f"Empty Arabic translation for key '{k}'"

    for k, v in en_data.items():
        assert isinstance(v, str) and len(v.strip()) > 0, f"Empty English translation for key '{k}'"

"""Self-checks for the code-only metric fix and the translated-code filter.
Run: python test_fixes.py   (or pytest)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "eval"), os.path.join(HERE, "data")]
from metrics import score, summarize  # noqa: E402
from prepare_data import code_translated  # noqa: E402


def test_code_only_reference_is_not_counted_as_non_korean():
    code = "```sql\nSELECT email FROM Users;\n```"
    rows = [score(code, code), score("리스트는 sorted()로 정렬합니다.", "sorted()로 정렬합니다.")]
    s = summarize(rows)
    assert s["korean_cases"] == 1 and s["korean_answers_pct"] == 100.0


def test_translated_code_is_detected():
    assert code_translated("```python\n에서 langdetect import detect\n```")
    assert code_translated("SELECT * FROM 주택에서 도시 = '뉴욕시';")
    assert code_translated("'''파이썬\nprint(1)\n'''")
    assert not code_translated("```python\nname = '이메일'  # 한글 주석\nprint(name)\n```")


if __name__ == "__main__":
    test_code_only_reference_is_not_counted_as_non_korean()
    test_translated_code_is_detected()
    print("ok")

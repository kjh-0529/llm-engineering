"""Metrics for Korean answers from a small LLM.

- korean_ratio: share of Hangul among Hangul + Latin letters, outside code
  (fenced blocks and `inline code`). Technical terms in English lower it a
  little; whole English answers drive it toward 0. Reported only for items
  whose reference answer is itself Korean: a correct code-only answer has no
  letters outside code and would otherwise count as "not Korean".
- cjk_drift: the answer contains Chinese characters or Japanese kana outside
  code (3 or more). Small multilingual models often drift into Chinese when
  asked in Korean.
- chrf: character n-gram F-score against the reference answer (sacrebleu chrF).
  Character-level, so it suits Korean, where spacing and particles vary.
- code_ok: if the reference contains a fenced code block, the answer contains
  one too, and its blocks keep their line breaks.
"""
import re

from sacrebleu.metrics import CHRF

CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
FENCED = re.compile(r"```(.*?)```", re.DOTALL)
HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
LATIN = re.compile(r"[A-Za-z]")
HAN_KANA = re.compile(r"[一-鿿぀-ヿ]")
_chrf = CHRF()


def strip_code(text):
    return CODE.sub(" ", text)


def korean_ratio(text):
    t = strip_code(text)
    h, l = len(HANGUL.findall(t)), len(LATIN.findall(t))
    return h / (h + l) if h + l else 0.0


def cjk_drift(text):
    return len(HAN_KANA.findall(strip_code(text))) >= 3


def chrf(answer, reference):
    return _chrf.sentence_score(answer, [reference]).score


def code_ok(answer, reference):
    if "```" not in reference:
        return None  # not applicable
    blocks = FENCED.findall(answer)
    return bool(blocks) and all("\n" in b.strip("\n ") or len(b) < 40 for b in blocks)


def score(answer, reference):
    return {
        "korean_ratio": korean_ratio(answer),
        "ref_korean": korean_ratio(reference) >= 0.5,
        "cjk_drift": cjk_drift(answer),
        "chrf": chrf(answer, reference),
        "code_ok": code_ok(answer, reference),
        "chars": len(answer),
    }


def summarize(rows):
    n = len(rows)
    code = [r["code_ok"] for r in rows if r["code_ok"] is not None]
    kor = [r for r in rows if r["ref_korean"]]
    k = len(kor) or 1
    return {
        "n": n,
        "korean_cases": len(kor),
        "korean_ratio_mean": round(sum(r["korean_ratio"] for r in kor) / k, 4),
        "korean_answers_pct": round(100 * sum(r["korean_ratio"] >= 0.5 for r in kor) / k, 1),
        "cjk_drift_pct": round(100 * sum(r["cjk_drift"] for r in rows) / n, 1),
        "chrf_mean": round(sum(r["chrf"] for r in rows) / n, 2),
        "code_ok_pct": round(100 * sum(code) / len(code), 1) if code else None,
        "code_cases": len(code),
        "mean_chars": round(sum(r["chars"] for r in rows) / n),
    }

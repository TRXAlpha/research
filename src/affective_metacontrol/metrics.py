"""Behavioral, linguistic, and cognitive quality metrics for steered LLM outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
from typing import Sequence


def tokenize_words(text: str) -> list[str]:
    """Extract lowercase word tokens from text."""
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Return sequence of n-grams."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def n_gram_repetition_rate(tokens: Sequence[str], n: int = 3) -> float:
    """Calculate the fraction of non-unique n-grams in the token sequence.

    Returns 0.0 if all n-grams are unique, approaching 1.0 if identical n-grams repeat.
    """
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    unique_count = len(set(ngrams))
    return max(0.0, 1.0 - (unique_count / len(ngrams)))


def distinct_n(tokens: Sequence[str], n: int) -> float:
    """Distinct-n metric: unique n-grams divided by total n-grams."""
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def type_token_ratio(tokens: Sequence[str]) -> float:
    """Type-Token Ratio (TTR): unique vocabulary size divided by total token count."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def token_entropy(tokens: Sequence[str]) -> float:
    """Shannon entropy of unigram token distribution in bits."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def detect_repeated_phrase_loop(tokens: Sequence[str], min_phrase_len: int = 3, min_repeats: int = 3) -> bool:
    """Detect if a phrase of length >= min_phrase_len repeats consecutively >= min_repeats times."""
    n = len(tokens)
    for length in range(min_phrase_len, min(12, n // min_repeats + 1)):
        for start in range(n - length * min_repeats + 1):
            phrase = tokens[start : start + length]
            consecutive = 1
            for step in range(1, min_repeats):
                next_phrase = tokens[start + step * length : start + (step + 1) * length]
                if next_phrase == phrase:
                    consecutive += 1
                else:
                    break
            if consecutive >= min_repeats:
                return True
    return False


@dataclass(frozen=True, slots=True)
class TextQualityMetrics:
    """Comprehensive linguistic coherence and lexical diversity metrics."""

    token_count: int
    ttr: float
    distinct_1: float
    distinct_2: float
    distinct_3: float
    repetition_rate_2: float
    repetition_rate_3: float
    entropy: float
    is_degenerate_loop: bool
    coherence_log_prob: float | None = None

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def evaluate_text_quality(
    text: str,
    *,
    coherence_scorer=None,
) -> TextQualityMetrics:
    """Compute lexical diversity, repetition, loop detection, and optional model coherence."""
    tokens = tokenize_words(text)
    token_count = len(tokens)

    ttr = type_token_ratio(tokens)
    d1 = distinct_n(tokens, 1)
    d2 = distinct_n(tokens, 2)
    d3 = distinct_n(tokens, 3)
    rep2 = n_gram_repetition_rate(tokens, 2)
    rep3 = n_gram_repetition_rate(tokens, 3)
    ent = token_entropy(tokens)
    loop = detect_repeated_phrase_loop(tokens)

    coherence = None
    if coherence_scorer is not None and token_count > 1:
        coherence = float(coherence_scorer(text))

    return TextQualityMetrics(
        token_count=token_count,
        ttr=ttr,
        distinct_1=d1,
        distinct_2=d2,
        distinct_3=d3,
        repetition_rate_2=rep2,
        repetition_rate_3=rep3,
        entropy=ent,
        is_degenerate_loop=loop,
        coherence_log_prob=coherence,
    )


def compute_recovery_ratio(
    baseline_val: float,
    peak_val: float,
    current_val: float,
) -> float:
    """Measure how much of the peak disturbance has recovered back to baseline.

    0.0 = still at peak disturbance.
    1.0 = fully recovered to baseline.
    """
    total_disturbance = abs(peak_val - baseline_val)
    if total_disturbance < 1e-6:
        return 1.0
    recovered = total_disturbance - abs(current_val - baseline_val)
    return max(0.0, min(1.0, recovered / total_disturbance))

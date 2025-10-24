from __future__ import annotations

from collections import Counter
from typing import List


def bag_of_words_counts(tokens: List[str]) -> dict:
    cnt = Counter([t.lower() for t in tokens])
    return dict(cnt)

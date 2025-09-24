# spell_corrector.py (replace existing)
from rapidfuzz import process, fuzz
from typing import List

class SpellCorrector:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: list of known words (tokens).
        """
        # ensure tokens are normalized
        self.vocab = [v.strip().lower() for v in set(vocabulary) if v and str(v).strip()]
        # rapidfuzz works with sequences; store as list
        self._vocab = self.vocab

    def correct_word(self, word: str, threshold: int = 80) -> str:
        if not word:
            return word
        word = str(word).strip().lower()
        match = process.extractOne(word, self._vocab, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            return match[0]
        return word  # return normalized original if no close match

    def correct_query(self, query: str, threshold: int = 80) -> str:
        if not query:
            return query
        words = [w for w in str(query).split() if w.strip()]
        corrected = [self.correct_word(w, threshold=threshold) for w in words]
        return " ".join(corrected)

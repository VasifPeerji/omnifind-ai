from rapidfuzz import process, fuzz
from typing import List

class SpellCorrector:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: list of known words (brands, categories, product tokens).
        """
        self.vocab = list(set(vocabulary))  # unique words

    def correct_word(self, word: str, threshold: int = 80) -> str:
        """
        Correct a single word if a close match is found in vocab.
        """
        match = process.extractOne(word, self.vocab, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            return match[0]
        return word  # return as-is if no close match

    def correct_query(self, query: str) -> str:
        """
        Corrects each word in a query.
        """
        words = query.split()
        corrected = [self.correct_word(w) for w in words]
        return " ".join(corrected)

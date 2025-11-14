# src/omnifind/utils/spell_corrector.py
"""
Context-aware spell correction that protects common words.
Prevents: "for"→"fore", "women"→"womena", "wear"→"ear"
"""
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
from typing import List, Set

class SpellCorrector:
    """Conservative spell correction for product search"""
    
    # Common words that should NEVER be corrected
    PROTECTED_WORDS = {
        # Prepositions & articles
        'for', 'with', 'in', 'on', 'of', 'the', 'and', 'or', 'a', 'an',
        # Gender/age
        'men', 'women', 'man', 'woman', 'boy', 'girl', 'kid', 'kids',
        'male', 'female', 'boys', 'girls', 'ladies', 'mens', 'womens',
        # Common descriptors
        'new', 'old', 'big', 'small', 'large', 'medium',
        'wear', 'shirt', 'shoes', 'dress', 'pants', 'jacket',
        # Colors
        'red', 'blue', 'black', 'white', 'green', 'pink', 'gray', 'grey',
        'navy', 'brown', 'yellow', 'purple', 'orange', 'silver', 'gold',
        # Sizes
        's', 'm', 'l', 'xl', 'xxl', 'xs', 'xxs', 'xxxl',
        # Common actions
        'buy', 'get', 'find', 'cheap', 'best', 'top', 'good',
    }
    
    def __init__(self, vocabulary: List[str], min_word_length: int = 4):
        """
        Args:
            vocabulary: Known product tokens from your dataset
            min_word_length: Only correct words >= this length (default: 4)
        """
        # Normalize vocabulary
        self.vocab = [v.strip().lower() for v in set(vocabulary) if v and str(v).strip()]
        self._vocab_set = set(self.vocab)
        self.min_length = min_word_length
        
        print(f"[SpellCorrector] Loaded {len(self.vocab):,} vocabulary terms")
        print(f"[SpellCorrector] Min correction length: {min_word_length} chars")
    
    def correct_word(self, word: str, threshold: int = 85) -> str:
        """
        Correct single word with multiple safety checks.
        
        Args:
            word: Input word
            threshold: Similarity threshold (85 = strict, 70 = lenient)
        
        Returns:
            Corrected word or original if no safe correction found
        """
        if not word:
            return word
        
        word_lower = str(word).strip().lower()
        
        # SAFETY CHECK 1: Never correct protected words
        if word_lower in self.PROTECTED_WORDS:
            return word_lower
        
        # SAFETY CHECK 2: Never correct if already in vocabulary
        if word_lower in self._vocab_set:
            return word_lower
        
        # SAFETY CHECK 3: Only correct words >= min_length
        # (Prevents breaking short words like "for", "men", "ear")
        if len(word_lower) < self.min_length:
            return word_lower
        
        # SAFETY CHECK 4: Never correct words with digits
        # (Protects: "32gb", "B0979NG867", "4k", etc.)
        if any(c.isdigit() for c in word_lower):
            return word_lower
        
        # Search for correction
        match = process.extractOne(word_lower, self.vocab, scorer=fuzz.ratio)
        
        if match and match[1] >= threshold:
            corrected = match[0]
            
            # SAFETY CHECK 5: Prevent corrections with large edit distance
            # (Prevents: "women" → "womena" which has edit distance > 2)
            edit_dist = Levenshtein.distance(word_lower, corrected)
            if edit_dist > 2:
                return word_lower
            
            # SAFETY CHECK 6: Prevent corrections with large length difference
            # (Prevents: "for" → "fore" type errors)
            len_diff = abs(len(word_lower) - len(corrected))
            if len_diff > 2:
                return word_lower
            
            return corrected
        
        # No correction found - return original
        return word_lower
    
    def correct_query(self, query: str, threshold: int = 85) -> str:
        """
        Correct full query word-by-word.
        
        Args:
            query: User search query
            threshold: Similarity threshold (default: 85 for strict correction)
        
        Returns:
            Corrected query string
        """
        if not query:
            return query
        
        words = [w for w in str(query).split() if w.strip()]
        corrected = [self.correct_word(w, threshold=threshold) for w in words]
        return " ".join(corrected)
    
    def get_suggestions(self, word: str, top_n: int = 5) -> List[tuple]:
        """
        Get top N correction suggestions for debugging.
        
        Returns:
            List of (word, similarity_score) tuples
        """
        if not word:
            return []
        
        word_lower = str(word).strip().lower()
        matches = process.extract(word_lower, self.vocab, scorer=fuzz.ratio, limit=top_n)
        return matches


# === TESTING ===
if __name__ == "__main__":
    # Test vocabulary
    test_vocab = [
        "running", "shoes", "nike", "adidas", "black", "white",
        "women", "men", "dress", "shirt", "jacket", "pants",
        "cotton", "leather", "cheap", "expensive", "best",
        "forearm", "warehouse", "earring", "forest"  # Similar to common words
    ]
    
    corrector = SpellCorrector(test_vocab, min_word_length=4)
    
    # Test cases that were previously broken
    test_cases = [
        "for",        # Should NOT correct to "fore" or "forearm"
        "women",      # Should NOT correct to "womena"
        "wear",       # Should NOT correct to "ear" or "earring"
        "men",        # Should NOT correct
        "runnig",     # Should correct to "running"
        "shose",      # Should correct to "shoes"
        "addidas",    # Should correct to "adidas"
        "B0979NG867", # Should NOT correct (has digits)
    ]
    
    print("\nSpell Correction Tests:")
    print("=" * 60)
    for word in test_cases:
        corrected = corrector.correct_word(word)
        status = "✓ PROTECTED" if corrected == word.lower() else f"→ {corrected}"
        print(f"  '{word}' {status}")
    
    print("\n" + "=" * 60)
    print("Full Query Tests:")
    print("=" * 60)
    
    test_queries = [
        "nike runnig shoes for men",
        "women red dress",
        "addidas black shose",
    ]
    
    for query in test_queries:
        corrected = corrector.correct_query(query)
        print(f"  Original:  '{query}'")
        print(f"  Corrected: '{corrected}'")
        print()
# src/omnifind/retrieval/query_analyzer.py
"""
Amazon-style query understanding and routing.
Extracts brands, colors, attributes and determines optimal search strategy.
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class QueryIntent:
    """Parsed query structure"""
    original: str
    clean_query: str
    query_type: str  # 'asin', 'branded', 'attributed', 'generic'
    brand: Optional[str] = None
    colors: List[str] = None
    attributes: Dict[str, any] = None
    search_strategy: str = 'hybrid'  # 'exact', 'keyword', 'hybrid', 'semantic'
    alpha: float = 0.6  # Dynamic fusion weight
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = []
        if self.attributes is None:
            self.attributes = {}


class QueryAnalyzer:
    """
    Analyzes queries to extract structured info and determine search strategy.
    """
    
    # Extended brand patterns (FASHION-SPECIFIC for your dataset)
    BRANDS = {
        # Athletic/Sportswear
        'nike', 'adidas', 'puma', 'reebok', 'under armour', 'new balance',
        'asics', 'skechers', 'fila', 'converse', 'vans', 'champion',
        # Casual/Fast Fashion
        'levi', 'levis', "levi's", 'wrangler', 'lee', 'gap', 'old navy',
        'zara', 'h&m', 'hm', 'uniqlo', 'forever 21', 'mango', 'guess',
        # Premium/Designer
        'calvin klein', 'tommy hilfiger', 'ralph lauren', 'polo',
        'armani', 'versace', 'gucci', 'prada', 'diesel',
        # Outdoor/Workwear
        'north face', 'columbia', 'patagonia', 'carhartt', 'timberland', 'dickies',
        # Footwear Specialists
        'dr martens', 'clarks', 'crocs', 'birkenstock',
        # Watches & Accessories
        'casio', 'fossil', 'timex', 'seiko', 'citizen', 'michael kors', 'skagen', 'bulova',
        # Plus common misspellings
        'addidas', 'calvinklein', 'tommyhilfiger', 'ralphlauren',
    }
    
    # Colors with common misspellings
    COLORS = {
        'black', 'white', 'blue', 'red', 'green', 'yellow', 'pink',
        'purple', 'orange', 'gray', 'grey', 'brown', 'beige', 'tan',
        'navy', 'maroon', 'burgundy', 'olive', 'khaki', 'cream',
        'silver', 'gold', 'rose gold', 'bronze', 'copper',
        'teal', 'turquoise', 'mint', 'lime', 'coral', 'lavender',
        # Color combinations
        'multi color', 'multicolor', 'multi-color',
        'two tone', 'two-tone'
    }
    
    # ASIN pattern (Amazon Standard Identification Number)
    ASIN_PATTERN = re.compile(r'\b(B[0-9A-Z]{9})\b', re.IGNORECASE)
    
    # Attribute patterns
    GENDER_PATTERNS = {
        'men': r'\b(men|mens|man|male|boys?)\b',
        'women': r'\b(women|womens|woman|female|girls?|ladies)\b',
        'kids': r'\b(kids?|children|toddler|infant|baby)\b',
        'unisex': r'\bunisex\b'
    }
    
    SIZE_PATTERN = re.compile(
        r'\b(x{0,3}[sml]|small|medium|large|[\d]+(?:\.\d+)?\s*(?:inch|in|cm|mm|oz|lb|kg|ml|l|gb|tb))\b',
        re.IGNORECASE
    )
    
    MATERIAL_PATTERNS = [
        'cotton', 'polyester', 'leather', 'suede', 'denim', 'wool',
        'silk', 'satin', 'nylon', 'spandex', 'mesh', 'canvas',
        'rubber', 'plastic', 'metal', 'wood', 'glass', 'ceramic'
    ]
    
    def __init__(self):
        # Compile patterns for performance
        self._brand_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(b) for b in self.BRANDS) + r')\b',
            re.IGNORECASE
        )
        self._color_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(c) for c in self.COLORS) + r')\b',
            re.IGNORECASE
        )
        self._material_pattern = re.compile(
            r'\b(' + '|'.join(self.MATERIAL_PATTERNS) + r')\b',
            re.IGNORECASE
        )
    
    def analyze(self, query: str) -> QueryIntent:
        """
        Main analysis function.
        Returns structured query intent with optimal search strategy.
        """
        query = query.strip()
        
        # Check for ASIN (exact match takes priority)
        asin_match = self.ASIN_PATTERN.search(query)
        if asin_match:
            return QueryIntent(
                original=query,
                clean_query=asin_match.group(1).upper(),
                query_type='asin',
                search_strategy='exact',  # Use BM25 only
                alpha=0.0,  # 100% keyword search
                attributes={'asin': asin_match.group(1).upper()}
            )
        
        # Extract structured attributes
        brand = self._extract_brand(query)
        colors = self._extract_colors(query)
        gender = self._extract_gender(query)
        size = self._extract_size(query)
        material = self._extract_material(query)
        
        # Build clean query - KEEP brand in query for semantic understanding
        clean_query = query.lower()
        
        # CRITICAL FIX: Don't remove brand from query!
        # Brand pre-filtering handles ensuring correct brand
        # Keeping brand in query helps semantic search understand context
        # Example: "converse shoes" needs "converse" for proper embeddings
        
        # Determine query type and strategy
        attributes = {}
        if brand:
            attributes['brand'] = brand
        if colors:
            attributes['colors'] = colors
        if gender:
            attributes['gender'] = gender
        if size:
            attributes['size'] = size
        if material:
            attributes['material'] = material
        
        # Enhanced query classification
        # FIXED: Better alpha values for brand queries
        if brand and (colors or gender):
            query_type = 'attributed'
            search_strategy = 'hybrid'
            alpha = 0.45  # Balanced: brand filter + semantic product matching
        elif brand:
            query_type = 'branded'
            search_strategy = 'hybrid'
            alpha = 0.5  # FIXED: More semantic for brand-only queries
        elif colors or gender or size:
            query_type = 'attributed'
            search_strategy = 'hybrid'
            alpha = 0.5  # Balanced
        else:
            query_type = 'generic'
            search_strategy = 'semantic'
            alpha = 0.7  # 70% semantic, 30% keyword
        
        # Clean up query (only whitespace, keep all words)
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
        return QueryIntent(
            original=query,
            clean_query=clean_query if clean_query else query.lower(),
            query_type=query_type,
            brand=brand,
            colors=colors,
            attributes=attributes,
            search_strategy=search_strategy,
            alpha=alpha
        )
    
    def _extract_brand(self, query: str) -> Optional[str]:
        """Extract brand name"""
        match = self._brand_pattern.search(query)
        if match:
            brand = match.group(1).lower()
            # Normalize variations
            if brand in ("levi's", 'levis'):
                return 'levi'
            if brand == 'hm':
                return 'h&m'
            return brand
        return None
    
    def _extract_colors(self, query: str) -> List[str]:
        """Extract all color mentions"""
        matches = self._color_pattern.findall(query)
        if matches:
            # Normalize
            colors = []
            for m in matches:
                c = m.lower()
                if c == 'grey':
                    c = 'gray'
                colors.append(c)
            return list(set(colors))  # Deduplicate
        return []
    
    def _extract_gender(self, query: str) -> Optional[str]:
        """Extract target gender"""
        for gender, pattern in self.GENDER_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                return gender
        return None
    
    def _extract_size(self, query: str) -> Optional[str]:
        """Extract size information"""
        match = self.SIZE_PATTERN.search(query)
        return match.group(0).lower() if match else None
    
    def _extract_material(self, query: str) -> Optional[str]:
        """Extract material"""
        match = self._material_pattern.search(query)
        return match.group(1).lower() if match else None


# === TESTING ===
if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    
    test_queries = [
        "adidas running shoes",
        "nike black shoes for men",
        "red dress women",
        "B0979NG867",
        "blue denim jacket",
        "running shoes under $50",
        "laptop 16gb ram",
        "cotton t-shirt large",
    ]
    
    print("Query Analysis Results:")
    print("=" * 80)
    
    for q in test_queries:
        intent = analyzer.analyze(q)
        print(f"\nQuery: '{q}'")
        print(f"  Type: {intent.query_type}")
        print(f"  Strategy: {intent.search_strategy} (alpha={intent.alpha})")
        print(f"  Clean: '{intent.clean_query}'")
        if intent.brand:
            print(f"  Brand: {intent.brand}")
        if intent.colors:
            print(f"  Colors: {intent.colors}")
        if intent.attributes:
            print(f"  Attributes: {intent.attributes}")
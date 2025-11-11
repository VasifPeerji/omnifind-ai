# src/omnifind/evaluation/evaluate_retriever.py
"""
Evaluation framework for measuring retriever accuracy.
Metrics: MRR, nDCG@K, Recall@K, Precision@K
"""
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
import pandas as pd

class RetrieverEvaluator:
    def __init__(self, test_queries_path: str):
        """
        Load test queries from JSON file.
        
        Format:
        [
            {
                "query": "nike running shoes",
                "relevant_asins": ["B0979NG867", "B0BKLSC624"],
                "category": "Men's Shoes"
            },
            ...
        ]
        """
        with open(test_queries_path) as f:
            self.test_data = json.load(f)
        print(f"Loaded {len(self.test_data)} test queries")
    
    def mean_reciprocal_rank(self, results: List[str], relevant: List[str]) -> float:
        """
        MRR: 1 / rank of first relevant result.
        Higher is better (max = 1.0).
        """
        for i, asin in enumerate(results, 1):
            if asin in relevant:
                return 1.0 / i
        return 0.0
    
    def ndcg_at_k(self, results: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain.
        Measures ranking quality with position weighting.
        """
        dcg = sum(
            1 / np.log2(i + 2) 
            for i, asin in enumerate(results[:k]) 
            if asin in relevant
        )
        
        # Ideal DCG (all relevant items at top)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def recall_at_k(self, results: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Recall@K: % of relevant items found in top K.
        """
        found = sum(1 for asin in results[:k] if asin in relevant)
        return found / len(relevant) if relevant else 0.0
    
    def precision_at_k(self, results: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Precision@K: % of top K that are relevant.
        """
        found = sum(1 for asin in results[:k] if asin in relevant)
        return found / min(k, len(results)) if results else 0.0
    
    def evaluate(self, retriever, top_k: int = 10, verbose: bool = True) -> Dict[str, float]:
        """
        Run full evaluation suite.
        
        Args:
            retriever: Instance with search_text(query, top_k) method
            top_k: Number of results to retrieve
        
        Returns:
            Dict of metric scores
        """
        mrr_scores = []
        ndcg_scores = []
        recall_scores = []
        precision_scores = []
        latencies = []
        
        for test_case in self.test_data:
            query = test_case["query"]
            relevant_asins = test_case["relevant_asins"]
            
            # Search
            import time
            t0 = time.time()
            results, corrected, _ = retriever.search_text(query, top_k=top_k)
            latency = (time.time() - t0) * 1000
            latencies.append(latency)
            
            # Extract ASINs
            result_asins = [r.get("asin") or r.get("id") for r in results]
            result_asins = [a for a in result_asins if a]  # Remove None
            
            # Metrics
            mrr_scores.append(self.mean_reciprocal_rank(result_asins, relevant_asins))
            ndcg_scores.append(self.ndcg_at_k(result_asins, relevant_asins, k=top_k))
            recall_scores.append(self.recall_at_k(result_asins, relevant_asins, k=top_k))
            precision_scores.append(self.precision_at_k(result_asins, relevant_asins, k=top_k))
            
            if verbose:
                print(f"\nQuery: {query}")
                print(f"  MRR: {mrr_scores[-1]:.3f} | nDCG@{top_k}: {ndcg_scores[-1]:.3f}")
                print(f"  Relevant found: {sum(1 for a in result_asins if a in relevant_asins)}/{len(relevant_asins)}")
        
        metrics = {
            "MRR": np.mean(mrr_scores),
            f"nDCG@{top_k}": np.mean(ndcg_scores),
            f"Recall@{top_k}": np.mean(recall_scores),
            f"Precision@{top_k}": np.mean(precision_scores),
            "Avg_Latency_ms": np.mean(latencies),
            "P95_Latency_ms": np.percentile(latencies, 95),
        }
        
        return metrics
    
    def compare_retrievers(self, retriever_a, retriever_b, 
                          labels: Tuple[str, str] = ("A", "B"),
                          top_k: int = 10) -> pd.DataFrame:
        """
        A/B test two retrievers.
        
        Example:
            evaluator.compare_retrievers(
                old_retriever, 
                hybrid_retriever,
                labels=("FAISS-only", "Hybrid")
            )
        """
        print(f"\n{'='*60}")
        print(f"A/B Test: {labels[0]} vs {labels[1]}")
        print('='*60)
        
        print(f"\nEvaluating {labels[0]}...")
        metrics_a = self.evaluate(retriever_a, top_k=top_k, verbose=False)
        
        print(f"\nEvaluating {labels[1]}...")
        metrics_b = self.evaluate(retriever_b, top_k=top_k, verbose=False)
        
        # Compare
        df = pd.DataFrame({
            labels[0]: metrics_a,
            labels[1]: metrics_b,
        })
        df["Δ"] = df[labels[1]] - df[labels[0]]
        df["Δ%"] = (df["Δ"] / df[labels[0]]) * 100
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(df.to_string())
        
        # Winner
        winner = labels[1] if metrics_b["MRR"] > metrics_a["MRR"] else labels[0]
        improvement = abs(df.loc["MRR", "Δ%"])
        print(f"\n🏆 Winner: {winner} (+{improvement:.1f}% MRR)")
        
        return df


# === Generate sample test data ===
def generate_test_queries(products_file: str, output_file: str, n_queries: int = 100):
    """
    Generate test queries from product data.
    Use high-rated products with reviews as ground truth.
    """
    with open(products_file) as f:
        products = json.load(f)
    
    # Filter quality products
    quality = [
        p for p in products 
        if p.get("stars", 0) >= 4.0 and p.get("reviews", 0) >= 20
    ]
    
    # Sample
    import random
    random.seed(42)
    sampled = random.sample(quality, min(n_queries, len(quality)))
    
    test_queries = []
    for p in sampled:
        title = p.get("title", "")
        asin = p.get("asin", "")
        category = p.get("category_name", "")
        
        # Generate query (first 3-5 words of title)
        words = title.split()[:random.randint(3, 5)]
        query = " ".join(words)
        
        test_queries.append({
            "query": query,
            "relevant_asins": [asin],
            "category": category,
        })
    
    with open(output_file, "w") as f:
        json.dump(test_queries, f, indent=2)
    
    print(f"Generated {len(test_queries)} test queries → {output_file}")


# === CLI ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "evaluate", "compare"], required=True)
    parser.add_argument("--products", default="data/embeddings/products.json")
    parser.add_argument("--test-queries", default="data/evaluation/test_queries.json")
    parser.add_argument("--n-queries", type=int, default=100)
    args = parser.parse_args()
    
    if args.mode == "generate":
        Path("data/evaluation").mkdir(parents=True, exist_ok=True)
        generate_test_queries(args.products, args.test_queries, args.n_queries)
    
    elif args.mode == "evaluate":
        from omnifind.retrieval.hybrid_retriever_v2 import HybridRetriever
        
        retriever = HybridRetriever(alpha=0.6)
        evaluator = RetrieverEvaluator(args.test_queries)
        metrics = evaluator.evaluate(retriever, top_k=10)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for k, v in metrics.items():
            print(f"{k:20s}: {v:.4f}")
    
    elif args.mode == "compare":
        from omnifind.retrieval.retriever import RetrieverService
        from omnifind.retrieval.hybrid_retriever_v2 import HybridRetriever
        
        old = RetrieverService()
        new = HybridRetriever(alpha=0.6)
        
        evaluator = RetrieverEvaluator(args.test_queries)
        evaluator.compare_retrievers(old, new, labels=("FAISS-only", "Hybrid"))
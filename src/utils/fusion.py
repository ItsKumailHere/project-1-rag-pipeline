from collections import defaultdict
from typing import List, Tuple
from src.models.schemas import SearchResult

def reciprocal_rank_fusion(
    rankings: List[List[SearchResult]], 
    k: int = 60,
    top_n: int = 10
) -> List[SearchResult]:
    """
    Fuses multiple ranked lists using Reciprocal Rank Fusion (RRF).
    This is Dave Ebbelaar's pattern for combining BM25 and Dense retrieval.
    """
    scores = defaultdict(float)
    # Keep track of the original SearchResult objects
    chunk_map = {}
    
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] += 1.0 / (k + rank)
            chunk_map[result.chunk_id] = result
            
    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Return the top N fused results, updating their score to the RRF score
    fused_results = []
    for chunk_id in sorted_ids[:top_n]:
        orig = chunk_map[chunk_id]
        fused_results.append(
            SearchResult(
                chunk_id=orig.chunk_id,
                content=orig.content,
                score=scores[chunk_id],
                metadata=orig.metadata
            )
        )
        
    return fused_results

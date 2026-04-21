import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment


def jaccard_similarity(set_a, set_b) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def column_pair_similarity(col_q: Dict, col_c: Dict) -> float:
    """
    Compute similarity between two columns for union matching.

    Based on value overlap (Jaccard), cardinality similarity, type match
    """
    type_q = col_q.get('inferred_type', 'string')
    type_c = col_c.get('inferred_type', 'string')

    # Different types → no union compatibility
    if type_q != type_c:
        return 0.0

    vals_q = col_q.get('top_values', set())
    vals_c = col_c.get('top_values', set())
    card_q = col_q.get('cardinality_ratio', 0.0)
    card_c = col_c.get('cardinality_ratio', 0.0)

    # Value overlap (Jaccard)
    jaccard_sim = jaccard_similarity(vals_q, vals_c)

    # Cardinality similarity
    card_sim = 1.0 - abs(card_q - card_c)

    # Combined (weighted)
    return 0.7 * jaccard_sim + 0.3 * card_sim


def compute_unionability_score(profile_q: List[Dict], profile_c: List[Dict]) -> float:
    """
    Compute unionability using optimal column alignment (Hungarian algorithm).

    Returns: mean of best column pair similarities × coverage penalty [0, 1]
    """
    n_q = len(profile_q)
    n_c = len(profile_c)

    if n_q == 0 or n_c == 0:
        return 0.0

    # Build similarity matrix
    similarity_matrix = np.zeros((n_q, n_c))
    for i, col_q in enumerate(profile_q):
        for j, col_c in enumerate(profile_c):
            similarity_matrix[i, j] = column_pair_similarity(col_q, col_c)

    # Find optimal alignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    # Compute union score
    if len(row_ind) == 0:
        return 0.0

    aligned_scores = [similarity_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    mean_sim = np.mean(aligned_scores) if aligned_scores else 0.0

    coverage = min(n_q, n_c) / max(n_q, n_c)

    return float(mean_sim * coverage)

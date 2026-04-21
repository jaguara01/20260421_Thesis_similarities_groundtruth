from typing import List, Dict


def compute_joinability_score(profile_q: List[Dict], profile_c: List[Dict]) -> float:
    """
    Compute joinability as the maximum containment across all column pairs.

    For joining, we only need one column pair to have good overlap.
    Returns: max(containment) across all valid column pairs [0, 1]
    """
    max_score = 0.0
    numeric_cardinality_threshold = 0.8

    for col_q in profile_q:
        vals_q = col_q.get('top_values', set())
        type_q = col_q.get('inferred_type', 'string')
        card_q = col_q.get('cardinality_ratio', 0.0)

        # Skip high-cardinality numeric columns (not useful for joining)
        if type_q == 'numeric' and card_q > numeric_cardinality_threshold:
            continue

        if not vals_q:
            continue

        for col_c in profile_c:
            vals_c = col_c.get('top_values', set())
            type_c = col_c.get('inferred_type', 'string')
            card_c = col_c.get('cardinality_ratio', 0.0)

            # Skip high-cardinality numeric columns
            if type_c == 'numeric' and card_c > numeric_cardinality_threshold:
                continue

            if not vals_c:
                continue

            # check type of columns
            if type_q != type_c:
                continue

            # Compute containment (directional)
            # Containment: what fraction of set_a appears in set_b
            score_a = len(vals_q & vals_c) / len(vals_q)
            score_b = len(vals_c & vals_q) / len(vals_c)

            # Take max: either direction works for join
            max_score = max(max_score, score_a, score_b)

    return max_score

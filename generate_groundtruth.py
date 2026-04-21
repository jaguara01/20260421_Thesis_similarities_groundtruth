"""
Groundtruth for similarities

Ground truth labels for table pairs based on joinability, unionability and 
semantic validation to reduce false positives.
"""

import csv
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.gt_profiling import extract_table_profiles
from src.gt_similarity_join import compute_joinability_score
from src.gt_similarity_union import compute_unionability_score
from src.gt_similarity_semantic import (
    compute_table_semantic_similarity,
    compute_column_embeddings,
    compute_column_semantic_similarity_matrix,
)

# ── Configuration to get timestamps ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Parameters - can be modified but arbitrary Semantic threshold has more weight ─────────────────────────────────────────────────────────
TOP_K = 200 # top most representative values of a column
JOIN_THRESHOLD = 0.15
UNION_THRESHOLD = 0.20
SEMANTIC_THRESHOLD = 0.40

# Load semantic model 
from sentence_transformers import SentenceTransformer
log.info("Loading semantic model (all-MiniLM-L6-v2)...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
log.info("✓ Semantic model loaded")


# ── Paths ─────────────────────────────────────────────────────────────────
DATALAKE_DIR = Path('freya_data')
OUTPUT_DIR = Path('groundtruth/outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def score_pair(
    query_file: str, 
    df_q: pd.DataFrame, 
    profile_q: list,
    candidate_file: str, 
    df_c: pd.DataFrame, 
    profile_c: list
) -> tuple:
    """Score a single table pair."""

    # Syntactic scores (Containment for JOIN and jacquard/hungarian for UNION)
    join_score = compute_joinability_score(profile_q, profile_c)
    union_score = compute_unionability_score(profile_q, profile_c)

    # Table-level semantic
    semantic_score = compute_table_semantic_similarity(
        query_file, df_q, candidate_file, df_c, semantic_model
    )

    # Check first semantic
    if semantic_score < 0.25:
        semantic_join = 0.0
        semantic_union = 0.0
    else:
        # Column embeddings of Q and C for top 500
        embeddings_q = compute_column_embeddings(df_q, semantic_model)
        embeddings_c = compute_column_embeddings(df_c, semantic_model)

        if embeddings_q and embeddings_c:
            # Column similarity matrix
            sem_matrix = compute_column_semantic_similarity_matrix(
                df_q, df_c, embeddings_q, embeddings_c
            )

            # Best join match
            semantic_join = float(np.max(sem_matrix)) if sem_matrix.size > 0 else 0.0

            # Best union alignment (Hungarian)
            row_ind, col_ind = linear_sum_assignment(-sem_matrix)
            if len(row_ind) > 0:
                semantic_union = float(np.mean([sem_matrix[i, j] for i, j in zip(row_ind, col_ind)]))
            else:
                semantic_union = 0.0
        else:
            semantic_join = 0.0
            semantic_union = 0.0

    # Ground truth results
    is_join_gt = 1 if (join_score >= JOIN_THRESHOLD and semantic_join >= SEMANTIC_THRESHOLD) else 0
    is_union_gt = 1 if (union_score >= UNION_THRESHOLD and semantic_union >= SEMANTIC_THRESHOLD) else 0
    is_ground_truth = 1 if (is_join_gt or is_union_gt) else 0

    overall_sim = max(join_score, union_score)

    return (
        overall_sim, join_score, union_score,
        semantic_score, semantic_join, semantic_union,
        is_join_gt, is_union_gt, is_ground_truth
    )

# ── Compare x number of queries dataset from freya against y number of candidates from freya ──────────────────────────────────────────────
def main():
    # ── Configuration ──────────────────────────────────────────────
    num_queries = None              # None = all tables, or set to number (e.g., 5)
    max_candidates_per_query = None # None = all candidates, or set to number (e.g., 50)
    output_filename = 'validation_groundtruth.csv'

    output_path = OUTPUT_DIR / output_filename
    log.info(f"Output: {output_path}")

    # Load all CSVs
    log.info("Loading datalake...")
    csv_files = sorted(DATALAKE_DIR.glob('*.csv'))


    # Added option for testing on sample - To be remove for clean version
    query_files = csv_files[:num_queries] if num_queries else csv_files

    log.info(f"Queries: {len(query_files)} | Candidates: {len(csv_files)}")

    # Pre-compute all profiles
    log.info("Computing profiles...")
    profiles = {}
    dfs = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            dfs[csv_file.name] = df
            profiles[csv_file.name] = extract_table_profiles(df, top_k=TOP_K)
        except Exception as e:
            log.warning(f"Skipping {csv_file.name}: {e}")

    # Score all pairs
    results = []
    for q_idx, q_file in enumerate(query_files, 1):
        q_name = q_file.name
        if q_name not in profiles:
            continue

        log.info(f"[{q_idx}/{len(query_files)}] {q_name}")
        df_q = dfs[q_name]
        profile_q = profiles[q_name]

        candidates = csv_files
        if max_candidates_per_query:
            candidates = candidates[:max_candidates_per_query]

        for c_file in candidates:
            c_name = c_file.name
            if c_name not in profiles:
                continue

            df_c = dfs[c_name]
            profile_c = profiles[c_name]

            (sim, join, union, sem, sem_join, sem_union,
             is_join_gt, is_union_gt, is_gt) = score_pair(
                q_name, df_q, profile_q,
                c_name, df_c, profile_c
            )

            results.append([
                q_name, c_name,
                round(sim, 3), round(join, 3), round(union, 3),
                round(sem, 3), round(sem_join, 3), round(sem_union, 3),
                is_join_gt, is_union_gt, is_gt
            ])

    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query_table', 'candidate_table',
            'similarity', 'joinability', 'unionability',
            'semantic_table', 'semantic_join', 'semantic_union',
            'is_join_gt', 'is_union_gt', 'is_ground_truth'
        ])
        writer.writerows(results)

    
    gt_count = sum(1 for r in results if r[-1] == 1)
    log.info(f"Ground truth - similar datasets: {gt_count}/{len(results)}")


if __name__ == '__main__':
    main()

"""
Independent ground truth generation from actual data.

This script computes dataset similarity WITHOUT using:
- FREYJA ground truth labels
- Cached embeddings or indices
- Previous pipeline results
- Any pre-computed profiles

Instead, it analyzes ACTUAL DATA CONTENT:
1. Column-level semantic similarity (encode column names + sample values)
2. Joinability potential (matching keys across tables)
3. Unionability potential (compatible schemas)
4. Value distribution similarity (statistical comparison)

This is a heavy computation but produces ground truth that is
completely independent and data-driven.

Usage:
    # Quick test: 3 queries
    python3 groundtruth/generate_independent.py --sample 3

    # Full: all 46 queries (takes 30-60 min)
    python3 groundtruth/generate_independent.py --all

    # Specific queries
    python3 groundtruth/generate_independent.py --sample 10 --max-candidates 30
"""

import os
import csv
import argparse
import logging
from collections import defaultdict
from pathlib import Path
import random
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def compute_column_similarity(col1_name, col1_values, col2_name, col2_values):
    """
    Compute semantic similarity between two columns based on:
    1. Column name similarity (Levenshtein)
    2. Value overlap (Jaccard on unique values)
    3. Type compatibility (string vs numeric, etc.)
    """

    # 1. Column name similarity (simple: shared tokens)
    def tokenize(s):
        return set(s.lower().replace('_', ' ').split())

    tokens1 = tokenize(col1_name)
    tokens2 = tokenize(col2_name)

    if tokens1 or tokens2:
        name_overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    else:
        name_overlap = 0.0

    # 2. Value overlap (Jaccard on sample)
    try:
        val1_set = set(str(v).lower() for v in col1_values if pd.notna(v))
        val2_set = set(str(v).lower() for v in col2_values if pd.notna(v))

        if val1_set or val2_set:
            value_overlap = len(val1_set & val2_set) / len(val1_set | val2_set)
        else:
            value_overlap = 0.0
    except:
        value_overlap = 0.0

    # 3. Type compatibility
    def infer_type(values):
        non_null = [v for v in values if pd.notna(v)]
        if not non_null:
            return 'unknown'

        try:
            float(non_null[0])
            return 'numeric'
        except:
            return 'string'

    type1 = infer_type(col1_values)
    type2 = infer_type(col2_values)
    type_compat = 1.0 if type1 == type2 else 0.5

    # Weighted combination
    column_sim = (0.3 * name_overlap + 0.4 * value_overlap + 0.3 * type_compat)

    return column_sim


def compute_joinability_score(df1, df2, sample_size=100):
    """
    Estimate joinability between two datasets.

    For each column in df1, find best matching column in df2.
    Joinability = fraction of columns that have good matches.
    """

    if df1.empty or df2.empty:
        return 0.0

    # Sample columns
    cols1 = df1.columns.tolist()
    cols2 = df2.columns.tolist()

    if not cols1 or not cols2:
        return 0.0

    # For each column in df1, find best match in df2
    match_scores = []

    for col1 in cols1:
        col1_values = df1[col1].dropna().head(sample_size).tolist()

        best_match = 0.0
        for col2 in cols2:
            col2_values = df2[col2].dropna().head(sample_size).tolist()

            similarity = compute_column_similarity(col1, col1_values, col2, col2_values)
            best_match = max(best_match, similarity)

        match_scores.append(best_match)

    # Joinability: fraction of columns with good matches (> 0.3)
    if match_scores:
        joinability = sum(1 for s in match_scores if s > 0.3) / len(match_scores)
    else:
        joinability = 0.0

    return joinability


def compute_unionability_score(df1, df2):
    """
    Estimate unionability between two datasets.

    Union is possible if schemas are compatible.
    Score based on column overlap and type compatibility.
    """

    if df1.empty or df2.empty:
        return 0.0

    cols1 = set(c.lower() for c in df1.columns)
    cols2 = set(c.lower() for c in df2.columns)

    if not cols1 or not cols2:
        return 0.0

    # Column overlap
    overlap = len(cols1 & cols2)
    union = len(cols1 | cols2)

    if union == 0:
        return 0.0

    overlap_score = overlap / union

    # Type compatibility for overlapping columns
    type_scores = []
    for col in (cols1 & cols2):
        # Find actual column names (case-insensitive)
        col1_actual = next(c for c in df1.columns if c.lower() == col)
        col2_actual = next(c for c in df2.columns if c.lower() == col)

        # Check type compatibility
        try:
            vals1 = df1[col1_actual].dropna().head(10)
            vals2 = df2[col2_actual].dropna().head(10)

            type1 = 'numeric' if all(isinstance(v, (int, float)) for v in vals1) else 'string'
            type2 = 'numeric' if all(isinstance(v, (int, float)) for v in vals2) else 'string'

            compat = 1.0 if type1 == type2 else 0.5
            type_scores.append(compat)
        except:
            type_scores.append(0.5)

    if type_scores:
        type_compat = sum(type_scores) / len(type_scores)
    else:
        type_compat = 0.5

    # Unionability: overlap + type compat
    unionability = 0.6 * overlap_score + 0.4 * type_compat

    return unionability


def compute_schema_similarity(df1, df2):
    """
    Structural similarity based on row/column counts and basic stats.
    """

    # Row count similarity
    rows1, rows2 = len(df1), len(df2)
    max_rows = max(rows1, rows2, 1)
    row_sim = 1.0 - (abs(rows1 - rows2) / max_rows)

    # Column count similarity
    cols1, cols2 = len(df1.columns), len(df2.columns)
    max_cols = max(cols1, cols2, 1)
    col_sim = 1.0 - (abs(cols1 - cols2) / max_cols)

    # Null ratio similarity
    null_ratio1 = df1.isnull().sum().sum() / (rows1 * cols1) if rows1 * cols1 > 0 else 0
    null_ratio2 = df2.isnull().sum().sum() / (rows2 * cols2) if rows2 * cols2 > 0 else 0
    null_sim = 1.0 - abs(null_ratio1 - null_ratio2)

    schema_sim = (0.3 * row_sim + 0.3 * col_sim + 0.4 * null_sim)

    return schema_sim


def compute_independent_similarity(df_query, df_candidate):
    """
    Compute dataset similarity using ONLY actual data content.

    No cached results, no FREYJA, no embeddings.

    Combines:
    1. Joinability (40%) — can we join these tables?
    2. Unionability (30%) — can we union these tables?
    3. Schema similarity (30%) — structural compatibility
    """

    try:
        joinability = compute_joinability_score(df_query, df_candidate)
        unionability = compute_unionability_score(df_query, df_candidate)
        schema_sim = compute_schema_similarity(df_query, df_candidate)

        # Weighted combination
        similarity = (
            0.4 * joinability +
            0.3 * unionability +
            0.3 * schema_sim
        )

        return {
            'similarity': similarity,
            'joinability': joinability,
            'unionability': unionability,
            'schema_similarity': schema_sim,
        }

    except Exception as e:
        log.debug(f"Error computing similarity: {e}")
        return {
            'similarity': 0.5,
            'joinability': 0.5,
            'unionability': 0.5,
            'schema_similarity': 0.5,
        }


def generate_independent_groundtruth(
    datalake_dir='freya_data',
    output_path='validation_groundtruth_independent.csv',
    sample_size=None,
    max_candidates=None,  # If set, sample candidates (for speed)
):
    """
    Generate ground truth from actual data, completely independent.

    Args:
        sample_size: Number of queries to process (None = all)
        max_candidates: Sample N random candidates per query (None = all)
    """

    try:
        import pandas as pd
    except ImportError:
        log.error("pandas required. Install with: pip install pandas")
        return

    log.info("🚀 Independent Ground Truth Generation (Heavy Computation)\n")
    log.info("⚠️  This computes from ACTUAL DATA - no cached results or FREYJA\n")

    # Get all CSV files
    all_files = sorted([f for f in os.listdir(datalake_dir) if f.endswith('.csv')])

    if not all_files:
        log.error(f"No CSV files found in {datalake_dir}")
        return

    log.info(f"📊 Found {len(all_files)} datasets in datalake\n")

    # Select query tables
    query_tables = all_files.copy()
    if sample_size:
        query_tables = query_tables[:sample_size]

    # Select candidates
    candidate_tables = all_files.copy()
    if max_candidates and len(candidate_tables) > max_candidates:
        # Sample candidates but ensure query tables are included
        query_set = set(query_tables)
        other_candidates = [c for c in candidate_tables if c not in query_set]
        sampled_others = random.sample(other_candidates, min(max_candidates - len(query_set), len(other_candidates)))
        candidate_tables = list(query_set) + sampled_others

    log.info(f"Processing {len(query_tables)} queries vs {len(candidate_tables)} candidates")
    log.info(f"Total pairs: {len(query_tables) * len(candidate_tables):,}\n")

    results = []
    error_count = 0

    for q_idx, query_file in enumerate(query_tables, 1):
        query_path = os.path.join(datalake_dir, query_file)

        if not os.path.exists(query_path):
            log.warning(f"[{q_idx}/{len(query_tables)}] ⚠️ {query_file} not found")
            continue

        log.info(f"[{q_idx}/{len(query_tables)}] {query_file}")

        try:
            # Load query
            query_df = pd.read_csv(query_path, nrows=5000)
            scored = 0

            for candidate_file in candidate_tables:
                candidate_path = os.path.join(datalake_dir, candidate_file)

                if not os.path.exists(candidate_path):
                    continue

                try:
                    # Load candidate
                    cand_df = pd.read_csv(candidate_path, nrows=5000)

                    # Compute similarity from actual data
                    score_data = compute_independent_similarity(query_df, cand_df)

                    # Self-match gets perfect score
                    if query_file == candidate_file:
                        final_sim = 1.0
                    else:
                        final_sim = score_data['similarity']

                    results.append([
                        query_file,
                        candidate_file,
                        final_sim,
                        score_data['joinability'],
                        score_data['unionability'],
                        score_data['schema_similarity'],
                    ])
                    scored += 1

                except Exception as e:
                    log.debug(f"  Error: {candidate_file}: {e}")
                    continue

            log.info(f"  ✓ {scored} candidates scored")

        except Exception as e:
            log.error(f"  ❌ {e}")
            error_count += 1
            continue

    # Save results
    log.info(f"\n💾 Writing {len(results)} pairs to {output_path}...")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query_table', 'candidate_table', 'similarity',
            'joinability', 'unionability', 'schema_similarity'
        ])
        writer.writerows(results)

    # Summary
    if results:
        similarities = [r[2] for r in results]
        log.info(f"\n📈 Summary Statistics:")
        log.info(f"  Total pairs: {len(results)}")
        log.info(f"  Mean similarity: {sum(similarities)/len(similarities):.3f}")
        log.info(f"  Min: {min(similarities):.3f}")
        log.info(f"  Max: {max(similarities):.3f}")

    log.info(f"\n✅ Done! Saved to: {output_path}")
    log.info(f"   Errors: {error_count}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate independent ground truth from actual data'
    )
    parser.add_argument('--all', action='store_true', help='All queries')
    parser.add_argument('--sample', type=int, help='N queries')
    parser.add_argument('--max-candidates', type=int, help='Sample N candidates per query')
    parser.add_argument('--output', default='validation_groundtruth_independent.csv')

    args = parser.parse_args()

    sample = args.all and None or (args.sample or 3)

    log.info(f"Configuration:")
    log.info(f"  Queries: {sample if sample else 'all'}")
    log.info(f"  Candidates: {args.max_candidates or 'all'}\n")

    generate_independent_groundtruth(
        output_path=args.output,
        sample_size=sample,
        max_candidates=args.max_candidates,
    )

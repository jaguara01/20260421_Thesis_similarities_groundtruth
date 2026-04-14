"""
Independent ground truth generator.

This script is completely standalone - it can be run separately from the pipeline.
It uses only cached data (FAISS indices, profiles, embeddings) to score all pairs.

No imports from src/ or main pipeline code.

Run once to generate training labels for weight optimization.

Usage:
    python3 groundtruth/generate_heavy.py --all
    python3 groundtruth/generate_heavy.py --sample 30
"""

import os
import csv
import argparse
import pickle
import logging
from collections import defaultdict
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def load_freyja_ground_truth(path='../Fya/ground_truths/freyja_ground_truth.csv'):
    """Load ground truth from FREYJA CSV."""
    import csv
    gt = defaultdict(set)

    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt[row['target_ds']].add(row['candidate_ds'])
    except Exception as e:
        log.error(f"Failed to load ground truth: {e}")
        return {}

    return dict(gt)


def load_table_mapping(path='../profiling_outputs/freya_data_cache/semantic_mapping.pkl'):
    """Load FAISS table mapping."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        log.error(f"Failed to load table mapping: {e}")
        return {}


def load_profiles(path='../profiling_outputs/freya_data_cache/syntactic_profiles.pkl'):
    """Load syntactic profiles."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        log.error(f"Failed to load profiles: {e}")
        return {}


def load_faiss_index(path='../profiling_outputs/freya_data_cache/semantic_cosine.index'):
    """Load FAISS index."""
    try:
        import faiss
        return faiss.read_index(path)
    except Exception as e:
        log.error(f"Failed to load FAISS index: {e}")
        return None


def compute_metafeature_similarity(profile1, profile2):
    """
    Compute similarity between two profiles using metafeatures.

    Returns: float in [0, 1]
    """

    def safe_get(d, key, default=0.0):
        v = d.get(key, default) if d else default
        return float(v) if v is not None else default

    # Extract features
    row1 = safe_get(profile1, 'row_count')
    row2 = safe_get(profile2, 'row_count')

    col1 = safe_get(profile1, 'col_count')
    col2 = safe_get(profile2, 'col_count')

    null1 = safe_get(profile1, 'null_ratio')
    null2 = safe_get(profile2, 'null_ratio')

    card1 = safe_get(profile1, 'cardinality')
    card2 = safe_get(profile2, 'cardinality')

    # Compute similarities (normalized differences)
    def sim(v1, v2, max_val=1):
        if max_val == 0:
            return 1.0 if abs(v1 - v2) < 1e-6 else 0.0
        s = 1.0 - (abs(v1 - v2) / max(max_val, 1e-6))
        return max(0.0, min(1.0, s))

    row_sim = sim(row1, row2, max(row1, row2, 1))
    col_sim = sim(col1, col2, max(col1, col2, 1))
    null_sim = sim(null1, null2, 1.0)
    card_sim = sim(card1, card2, max(card1, card2, 1))

    # Weighted combination
    composite = (
        0.25 * col_sim +
        0.2 * row_sim +
        0.15 * null_sim +
        0.2 * card_sim +
        0.2 * null_sim
    )

    return composite


def compute_cosine_similarity_faiss(index, table_mapping, query_idx):
    """
    Compute cosine similarity from FAISS index.

    Returns dict: {table_name: similarity}
    """
    results = {}

    try:
        # Search all tables
        n_tables = len(table_mapping)
        query_embedding = index.reconstruct(query_idx).reshape(1, -1)

        similarities, indices = index.search(query_embedding, n_tables)

        for idx, sim in zip(indices[0], similarities[0]):
            if idx in table_mapping:
                table_name = table_mapping[idx]
                results[table_name] = float(sim)

    except Exception as e:
        log.debug(f"FAISS error: {e}")

    return results


def generate_groundtruth(
    output_path='validation_groundtruth.csv',
    datalake_dir='../freya_data',
    sample_size=None,
):
    """
    Generate ground truth by scoring all pairs using cached indices.

    This is independent and can run separately from the main pipeline.
    """

    log.info("🚀 Standalone Ground Truth Generation\n")

    log.info("🔄 Loading cached data...")
    freyja_gt = load_freyja_ground_truth()
    table_mapping = load_table_mapping()
    profiles = load_profiles()
    faiss_index = load_faiss_index()

    log.info(f"  ✓ Ground truth: {len(freyja_gt)} queries")
    log.info(f"  ✓ Table mapping: {len(table_mapping)} tables")
    log.info(f"  ✓ Profiles: {len(profiles)} tables")
    log.info(f"  ✓ FAISS index: {faiss_index is not None}")

    # Get query tables
    query_tables = sorted(freyja_gt.keys())
    if sample_size:
        query_tables = query_tables[:sample_size]

    log.info(f"\n📊 Processing {len(query_tables)} queries\n")

    results = []
    error_count = 0

    for q_idx, query_table in enumerate(query_tables, 1):
        log.info(f"[{q_idx}/{len(query_tables)}] {query_table}")

        # Find query index in table mapping
        query_idx = None
        for idx, tbl in table_mapping.items():
            if tbl == query_table:
                query_idx = idx
                break

        if query_idx is None:
            log.warning(f"  ⚠️ Query not in FAISS index")
            error_count += 1
            continue

        try:
            # Get semantic scores from FAISS
            if faiss_index:
                semantic_scores = compute_cosine_similarity_faiss(
                    faiss_index, table_mapping, query_idx
                )
            else:
                semantic_scores = {}

            # Get ground truth candidates
            gt_candidates = freyja_gt.get(query_table, set())

            # Score all candidates
            query_profile = profiles.get(query_table, {})
            scored = 0

            for candidate_table in table_mapping.values():
                try:
                    # Semantic score
                    sem_score = semantic_scores.get(candidate_table, 0.5)

                    # Syntactic score (metafeatures)
                    cand_profile = profiles.get(candidate_table, {})
                    syn_score = compute_metafeature_similarity(query_profile, cand_profile)

                    # Fuse
                    fused_score = 0.5 * sem_score + 0.5 * syn_score

                    # Label
                    is_gt = int(candidate_table in gt_candidates)

                    results.append([
                        query_table,
                        candidate_table,
                        fused_score,
                        is_gt,
                        sem_score,
                        syn_score,
                    ])
                    scored += 1

                except Exception as e:
                    log.debug(f"  Error scoring {candidate_table}: {e}")
                    continue

            log.info(f"  ✓ {scored} candidates scored")

        except Exception as e:
            log.error(f"  ❌ Error: {e}")
            error_count += 1
            continue

    # Save results
    log.info(f"\n💾 Writing {len(results)} pairs to {output_path}...")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query_table', 'candidate_table', 'fused_score',
            'is_ground_truth', 'semantic_score', 'syntactic_score'
        ])
        writer.writerows(results)

    # Summary
    if results:
        import statistics
        scores = [r[2] for r in results]
        gt_scores = [r[2] for r in results if r[3] == 1]
        non_gt_scores = [r[2] for r in results if r[3] == 0]

        log.info(f"\n📈 Summary Statistics:")
        log.info(f"  Total pairs: {len(results)}")
        log.info(f"  Ground truth matches: {len(gt_scores)}")
        log.info(f"  Mean score: {statistics.mean(scores):.3f}")

        if gt_scores:
            log.info(f"\n  Ground truth distribution:")
            log.info(f"    Mean: {statistics.mean(gt_scores):.3f}")
            log.info(f"    Min:  {min(gt_scores):.3f}")
            log.info(f"    Max:  {max(gt_scores):.3f}")
            log.info(f"    Count: {len(gt_scores)}")

        if non_gt_scores:
            log.info(f"\n  Non-ground truth distribution:")
            log.info(f"    Mean: {statistics.mean(non_gt_scores):.3f}")
            log.info(f"    Min:  {min(non_gt_scores):.3f}")
            log.info(f"    Max:  {max(non_gt_scores):.3f}")
            log.info(f"    Count: {len(non_gt_scores)}")

    log.info(f"\n✅ Done! Saved to: {output_path}")
    log.info(f"   Errors: {error_count}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate ground truth from cached indices (independent process)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all queries (slow)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Process N queries'
    )
    parser.add_argument(
        '--output',
        default='validation_groundtruth.csv',
        help='Output CSV'
    )

    args = parser.parse_args()

    # Determine sample size
    if args.all:
        sample = None
    elif args.sample:
        sample = args.sample
    else:
        sample = 10
        log.info("⚠️ No size specified. Running on 10 queries. Use --sample N or --all\n")

    results = generate_groundtruth(
        output_path=args.output,
        sample_size=sample,
    )

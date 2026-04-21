"""
Freyja-based ground truth generator for joinability and unionability.

Scores every (query, candidate) dataset pair using Freyja's column-level profiling:

  join_score   — max containment of top-10 frequent values across all column pairs.
                 Directional: measures whether a query column's key values appear in a
                 candidate column (equi-join opportunity).  ∈ [0, 1].

  union_score  — greedy schema alignment by column-name similarity, then Jaccard of
                 aligned column value sets, scaled by schema coverage.
                 Symmetric: both tables need similar schemas AND similar value domains.  ∈ [0, 1].

No external ground truth CSV is required.  Labels are self-supervised from data content.

Prerequisites:
    pip install faiss-cpu sentence-transformers  # already required by the pipeline
    # Freyja deps already in Fya/requirements.txt

Usage:
    # Quick test — 5 query datasets
    python3 groundtruth/generate_freyja_scores.py --sample 5

    # Full run — all datasets in freya_data/
    python3 groundtruth/generate_freyja_scores.py --all

    # Use Freyja's gradient-boosting model for join scoring (slower, more accurate)
    python3 groundtruth/generate_freyja_scores.py --all --use-model

    # Force rebuild of column profiles even if cache exists
    python3 groundtruth/generate_freyja_scores.py --all --rebuild-profiles

Output columns:
    query_table, candidate_table,
    join_score, union_score, combined_score,
    is_join_gt, is_union_gt, is_ground_truth
"""

import sys
import csv
import argparse
import logging
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FYA_PATH = _REPO_ROOT / 'Fya'
sys.path.insert(0, str(_FYA_PATH))

DEFAULT_DATALAKE     = _REPO_ROOT / 'freya_data'
DEFAULT_PROFILES_CSV = _REPO_ROOT / 'profiling_outputs' / 'column_profiles.csv'
# DataProfiler derives the pkl path by appending _normalized and swapping .csv → .pkl
DEFAULT_PROFILES_PKL = _REPO_ROOT / 'profiling_outputs' / 'column_profiles_normalized.pkl'
DEFAULT_MODEL        = _FYA_PATH / 'app' / 'core' / 'model' / 'gradient_boosting_ne100_lr0.05_md3_ss0.8_msl10.pkl'
DEFAULT_OUTPUT       = _REPO_ROOT / 'groundtruth_freyja_scores.csv'

JOIN_THRESHOLD  = 0.15  # min join_score  → is_join_gt  = 1
UNION_THRESHOLD = 0.20  # min union_score → is_union_gt = 1


# ── Primitive similarity functions ───────────────────────────────────────────

def containment(a: set, b: set) -> float:
    """Fraction of a's values that also appear in b.  Directional."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def jaccard(a: set, b: set) -> float:
    """Symmetric value overlap."""
    total = a | b
    if not total:
        return 0.0
    return len(a & b) / len(total)


def name_sim(n1: str, n2: str) -> float:
    """Token-level Jaccard on column names (lowercased, split on _ and -)."""
    def tokens(s):
        return set(s.lower().replace('_', ' ').replace('-', ' ').split())
    t1, t2 = tokens(n1), tokens(n2)
    total = t1 | t2
    if not total:
        return 0.0
    return len(t1 & t2) / len(total)


# ── Profile building / loading ────────────────────────────────────────────────

def build_or_load_profiles(
    datalake_dir: Path,
    profiles_pkl: Path,
    profiles_csv: Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load column-level profiles from cache, or build them via Freyja.

    The returned DataFrame has one row per column with fields:
        dataset_name, attribute_name,
        freq_word_containment        (set of top-10 frequent values, lowercased)
        freq_word_soundex_containment (set of soundex codes)
        cardinality, incompleteness, uniqueness, entropy,
        len_avg_word, len_min_word, len_max_word,
        frequency_avg, frequency_sd, ... (distribution metrics)
        first_word, last_word,  is_empty, is_binary
    """
    if profiles_pkl.exists() and not force_rebuild:
        log.info(f"Loading cached profiles: {profiles_pkl}")
        df = pd.read_pickle(profiles_pkl)
        log.info(f"  {len(df):,} columns across {df['dataset_name'].nunique()} datasets")
        return df

    log.info("Building column profiles via Freyja (first run — may take a few minutes)…")
    from app.core.profiling.profiler import DataProfiler, ProfilerConfig

    config = ProfilerConfig(
        datalake_path=datalake_dir,
        output_profiles_path=profiles_csv,  # DataProfiler derives normalized + pkl from this
        max_workers=8,
        varchar_only=False,  # include numeric cols — important for numeric key joins
    )
    profiler = DataProfiler(config)
    msg = profiler.generate_profiles_for_datalake()
    log.info(f"  {msg}")

    if not profiles_pkl.exists():
        raise FileNotFoundError(
            f"Freyja profiling completed but {profiles_pkl} was not created. "
            "Check DataProfiler logs above."
        )

    df = pd.read_pickle(profiles_pkl)
    log.info(f"  {len(df):,} columns across {df['dataset_name'].nunique()} datasets")
    return df


# ── Dataset-level join scoring ────────────────────────────────────────────────

def score_join_containment(cols_q: pd.DataFrame, cols_c: pd.DataFrame) -> float:
    """
    Join score based on max containment of frequent values.

    For each (query_column, candidate_column) pair:
      1. Primary signal: containment(freq_values_q, freq_values_c)
      2. Fallback:  0.7 × containment(soundex_q, soundex_c)  [handles misspellings]

    Returns max score across all pairs.  A single high-containment column pair
    is sufficient to declare the datasets joinable.
    """
    max_score = 0.0

    for _, rq in cols_q.iterrows():
        vals_q = rq.get('freq_word_containment', set())
        if not isinstance(vals_q, set) or not vals_q:
            continue
        sdx_q = rq.get('freq_word_soundex_containment', set())

        for _, rc in cols_c.iterrows():
            vals_c = rc.get('freq_word_containment', set())
            if not isinstance(vals_c, set) or not vals_c:
                continue

            score = containment(vals_q, vals_c)

            # Soundex fallback only when exact match is weak
            if score < 0.1 and isinstance(sdx_q, set) and sdx_q:
                sdx_c = rc.get('freq_word_soundex_containment', set())
                if isinstance(sdx_c, set):
                    score = max(score, 0.7 * containment(sdx_q, sdx_c))

            if score > max_score:
                max_score = score
                if max_score >= 1.0:
                    return 1.0  # early exit

    return max_score


def score_join_model(
    cols_q: pd.DataFrame,
    cols_c: pd.DataFrame,
    model,
    distance_patterns: dict,
) -> float:
    """
    Join score using Freyja's trained Gradient Boosting model.

    Replicates the feature construction in ComputeDistances, then applies the
    pre-trained model.  Returns max predicted score across all column pairs,
    normalized to [0, 1] using the 0.5 normalisation factor from ModelExecution.

    Use --use-model for higher accuracy at the cost of speed.
    """
    import Levenshtein as lev

    feature_names = (
        model.feature_names_in_
        if hasattr(model, 'feature_names_in_')
        else model.feature_names_
    )
    max_score = 0.0

    for _, rq in cols_q.iterrows():
        for _, rc in cols_c.iterrows():
            try:
                features = {}
                for col, pattern in distance_patterns.items():
                    v1, v2 = rq.get(col), rc.get(col)
                    if pattern == 'substraction':
                        v1 = float(v1) if pd.notna(v1) else 0.0
                        v2 = float(v2) if pd.notna(v2) else 0.0
                        features[col] = v1 - v2
                    elif pattern == 'containment':
                        s1 = v1 if isinstance(v1, set) else set()
                        s2 = v2 if isinstance(v2, set) else set()
                        features[col] = len(s1 & s2) / len(s1) if s1 else 0.0
                    elif pattern == 'levenshtein':
                        features[col] = lev.distance(
                            str(v1) if pd.notna(v1) else '',
                            str(v2) if pd.notna(v2) else '',
                        )

                features['name_dist'] = lev.distance(
                    str(rq.get('attribute_name', '')),
                    str(rc.get('attribute_name', '')),
                )

                row_df = pd.DataFrame([features])[feature_names]
                pred = float(model.predict(row_df)[0])
                max_score = max(max_score, min(pred / 0.5, 1.0))

            except Exception:
                continue

    return max_score


# ── Dataset-level union scoring ───────────────────────────────────────────────

def score_union(cols_q: pd.DataFrame, cols_c: pd.DataFrame) -> float:
    """
    Unionability score via greedy schema alignment + Jaccard value overlap.

    Algorithm:
      1. Build (n_q × n_c) name-similarity matrix.
      2. Greedy one-to-one assignment: for each query column (in order),
         assign the unmatched candidate column with the highest name similarity.
      3. For each assigned pair, compute Jaccard of freq_word_containment sets.
      4. union_score = mean(Jaccard) × schema_coverage
         where schema_coverage = n_aligned / max(n_q, n_c)

    The coverage term penalises tables with very different column counts,
    because a 2-column table always aligns well against a 50-column table
    without actually being unionable.
    """
    if cols_q.empty or cols_c.empty:
        return 0.0

    names_q = cols_q['attribute_name'].tolist()
    names_c = cols_c['attribute_name'].tolist()

    # Build name-similarity matrix
    sim_mat = np.zeros((len(names_q), len(names_c)))
    for i, nq in enumerate(names_q):
        for j, nc in enumerate(names_c):
            sim_mat[i, j] = name_sim(nq, nc)

    # Greedy one-to-one alignment (query drives the loop)
    assigned_c: set = set()
    aligned: list = []  # (idx_q, idx_c)

    for i in range(len(names_q)):
        best_j, best_s = -1, -1.0
        for j in range(len(names_c)):
            if j in assigned_c:
                continue
            if sim_mat[i, j] > best_s:
                best_s, best_j = sim_mat[i, j], j
        if best_j >= 0:
            assigned_c.add(best_j)
            aligned.append((i, best_j))

    if not aligned:
        return 0.0

    # Jaccard for each aligned pair
    jaccards = []
    for i, j in aligned:
        v_q = cols_q.iloc[i].get('freq_word_containment', set())
        v_c = cols_c.iloc[j].get('freq_word_containment', set())
        if not isinstance(v_q, set):
            v_q = set()
        if not isinstance(v_c, set):
            v_c = set()
        jaccards.append(jaccard(v_q, v_c))

    avg_jaccard = sum(jaccards) / len(jaccards)
    schema_coverage = len(aligned) / max(len(names_q), len(names_c))

    return avg_jaccard * schema_coverage


# ── Main generator ────────────────────────────────────────────────────────────

def generate(
    datalake_dir: Path = DEFAULT_DATALAKE,
    profiles_pkl: Path = DEFAULT_PROFILES_PKL,
    profiles_csv: Path = DEFAULT_PROFILES_CSV,
    output_path: Path = DEFAULT_OUTPUT,
    sample_size: Optional[int] = None,
    use_model: bool = False,
    rebuild_profiles: bool = False,
    join_threshold: float = JOIN_THRESHOLD,
    union_threshold: float = UNION_THRESHOLD,
) -> None:
    """
    Score all (query, candidate) pairs and write results to output_path.

    Skips self-pairs.  No external ground-truth CSV is used.
    """
    log.info("Freyja-based ground truth generation")
    log.info(f"  join_score  method : {'GB model' if use_model else 'containment (freq_word)'}")
    log.info(f"  union_score method : greedy schema alignment + Jaccard")
    log.info(f"  thresholds         : join>{join_threshold}, union>{union_threshold}\n")

    # ── 1. Load / build profiles ──────────────────────────────────────────────
    profiles = build_or_load_profiles(
        datalake_dir, profiles_pkl, profiles_csv, force_rebuild=rebuild_profiles
    )

    # Group by dataset for fast access
    grouped = {
        ds: grp.reset_index(drop=True)
        for ds, grp in profiles.groupby('dataset_name')
    }
    all_datasets = sorted(grouped.keys())
    log.info(f"Datasets available: {len(all_datasets)}")

    # ── 2. Optional: load GB model ────────────────────────────────────────────
    model = None
    distance_patterns = None
    if use_model:
        import joblib
        from app.core.profiling.metrics import MetricProperties
        log.info(f"Loading GB model from {DEFAULT_MODEL}")
        model = joblib.load(DEFAULT_MODEL)
        distance_patterns = MetricProperties().distance_patterns
        log.info("  Model loaded.")

    # ── 3. Select query tables ────────────────────────────────────────────────
    query_tables = all_datasets[:sample_size] if sample_size else all_datasets
    log.info(f"Queries: {len(query_tables)}  ×  Candidates: {len(all_datasets)}")
    log.info(f"Total pairs (excl. self): {len(query_tables) * (len(all_datasets) - 1):,}\n")

    # ── 4. Score all pairs ────────────────────────────────────────────────────
    rows = []
    for q_idx, q_ds in enumerate(query_tables, 1):
        log.info(f"[{q_idx}/{len(query_tables)}] {q_ds}")
        cols_q = grouped[q_ds]
        scored = 0

        for c_ds in all_datasets:
            if c_ds == q_ds:
                continue  # skip self-pair

            cols_c = grouped[c_ds]

            try:
                if use_model:
                    j_score = score_join_model(cols_q, cols_c, model, distance_patterns)
                else:
                    j_score = score_join_containment(cols_q, cols_c)

                u_score = score_union(cols_q, cols_c)
                combined = max(j_score, u_score)

                rows.append({
                    'query_table':    q_ds,
                    'candidate_table': c_ds,
                    'join_score':     round(j_score, 6),
                    'union_score':    round(u_score, 6),
                    'combined_score': round(combined, 6),
                    'is_join_gt':     int(j_score >= join_threshold),
                    'is_union_gt':    int(u_score >= union_threshold),
                    'is_ground_truth': int(
                        j_score >= join_threshold or u_score >= union_threshold
                    ),
                })
                scored += 1

            except Exception as e:
                log.debug(f"  skip {c_ds}: {e}")
                continue

        log.info(f"  {scored} pairs scored")

    # ── 5. Write output ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    log.info(f"\nWritten {len(df_out):,} rows → {output_path}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    if not df_out.empty:
        n_join  = df_out['is_join_gt'].sum()
        n_union = df_out['is_union_gt'].sum()
        n_gt    = df_out['is_ground_truth'].sum()
        total   = len(df_out)

        log.info(f"\nSummary")
        log.info(f"  Total pairs       : {total:,}")
        log.info(f"  is_join_gt  = 1   : {n_join:,}  ({100*n_join/total:.1f}%)")
        log.info(f"  is_union_gt = 1   : {n_union:,} ({100*n_union/total:.1f}%)")
        log.info(f"  is_ground_truth=1 : {n_gt:,}  ({100*n_gt/total:.1f}%)")
        log.info(f"\n  join_score   — mean {df_out['join_score'].mean():.3f}  "
                 f"max {df_out['join_score'].max():.3f}")
        log.info(f"  union_score  — mean {df_out['union_score'].mean():.3f}  "
                 f"max {df_out['union_score'].max():.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Self-supervised ground truth via Freyja column profiling'
    )
    parser.add_argument('--all', action='store_true',
                        help='Process all datasets as queries')
    parser.add_argument('--sample', type=int, metavar='N',
                        help='Process first N datasets as queries')
    parser.add_argument('--datalake', type=Path, default=DEFAULT_DATALAKE,
                        metavar='DIR')
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES_PKL,
                        metavar='PKL', help='Column profiles pickle (auto-built if missing)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        metavar='CSV')
    parser.add_argument('--use-model', action='store_true',
                        help='Use GB model for join scoring (slower, more principled)')
    parser.add_argument('--rebuild-profiles', action='store_true',
                        help='Force rebuild of column profiles from raw CSVs')
    parser.add_argument('--join-threshold', type=float, default=JOIN_THRESHOLD,
                        metavar='F')
    parser.add_argument('--union-threshold', type=float, default=UNION_THRESHOLD,
                        metavar='F')

    args = parser.parse_args()

    if not args.all and not args.sample:
        args.sample = 5
        log.info("No size specified — defaulting to --sample 5. Use --all for full run.\n")

    generate(
        datalake_dir=args.datalake,
        profiles_pkl=args.profiles,
        profiles_csv=DEFAULT_PROFILES_CSV,
        output_path=args.output,
        sample_size=None if args.all else args.sample,
        use_model=args.use_model,
        rebuild_profiles=args.rebuild_profiles,
        join_threshold=args.join_threshold,
        union_threshold=args.union_threshold,
    )

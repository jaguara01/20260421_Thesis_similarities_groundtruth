"""
Independent Ground Truth Generator (Enhanced with Semantic Validation)
=====================================================================
Scores every pair of datasets in the datalake for joinability and unionability
using full CSV content + semantic embeddings — no external labels, no pre-built indices.

Key Enhancement: Semantic Validation
-------------------------------------
To reduce false positives from coincidental value overlaps, this version integrates:
  1. Table-level semantic embeddings (using sentence-transformers)
  2. Multi-signal validation: ground truth requires BOTH:
     - Syntactic match (joinability ≥ 0.15 OR unionability ≥ 0.20)
     - Semantic support (embedding similarity ≥ 0.40)
  3. Full table processing (not just sampling) for more accurate profiles

What is "ground truth" here?
-----------------------------
In information retrieval, ground truth tells us which results are *correct*.
Here, a pair (query_table, candidate_table) is labelled as relevant if the
two datasets are either:
  - Joinable  : they share a common column whose values overlap enough to
                perform an equi-join (e.g. both have a "country" column with
                the same country names).
  - Unionable : their schemas are compatible — the columns cover the same
                domains so the rows could be stacked on top of each other
                (e.g. two sales tables from different years).

Why "independent"?
------------------
Other ground truth generators in this project (generate_heavy.py) rely on
pre-existing benchmark labels (freyja_ground_truth.csv) or pre-built FAISS
indices.  This script derives labels purely from data content, so it can be
applied to *any* datalake without prior annotation.

NEW: Semantic support ensures labels reflect actual similarity, not artifacts.

How are scores computed?
------------------------
Step 0 — Full Table Loading (NEW)
  Entire CSV files are loaded into memory (no sampling limit).
  This provides more accurate column signatures and enables semantic analysis.
  For very large tables (>1M rows), you can revert to sampling by setting MAX_ROWS.

Step 1 — Column Profile
  Each column is summarised into a compact signature from the FULL table:
  • top_values       : the 20 most frequent values (lowercased strings).
                       From value_counts() on entire column — distribution-aware.
  • inferred_type    : 'numeric' or 'string', inferred from actual values.
  • cardinality_ratio: fraction of rows with unique values (0 = constant
                       column, 1 = every row is different).
  • null_ratio       : fraction of missing values.
  • entropy          : Shannon entropy of the value distribution (bits).
                       High entropy → values are spread evenly.
                       Low entropy  → one or few values dominate.
  • numeric_stats    : (numeric columns only) min, max, mean, std, p25, p50,
                       p75, entropy — used for distribution-based comparison.

Step 2 — Semantic Similarity  [0, 1]  (NEW)
  Each table is serialized to text (table name, columns, types, sample values,
  sample rows) and embedded using sentence-transformers (all-MiniLM-L6-v2).
  Cosine similarity between embeddings captures semantic relatedness.
  Results are cached to avoid recomputation.

  Rationale: prevents false positives from coincidental value overlaps
  (e.g. two unrelated tables both having a "state" column with US state codes).

Step 3 — Joinability score  [0, 1]
  For each pair of columns (one from each table), compute the *containment*
  of top values: what fraction of table A's most common values appear in
  table B's most common values?
  The dataset join score is the MAXIMUM over all column pairs.
  Rationale: a single joinable column pair is enough to perform a join.

  High-cardinality numeric columns (e.g. prices, coordinates) are excluded
  because exact-value matching is meaningless for continuous data.

Step 4 — Unionability score  [0, 1]
  This uses a two-stage process that does NOT rely on column names:
  a) Build a similarity matrix between every pair of columns across the two
     tables, using:
     - String columns : Jaccard overlap of top-k values (55%) +
                        cardinality similarity (10%) +
                        entropy similarity (10%) + type match (25%)
     - Numeric columns: range overlap (40%) + mean proximity (25%) +
                        spread similarity (20%) + entropy similarity (15%)
     - Mixed types    : 0.0 (a string column cannot be unioned with a numeric)
  b) Find the OPTIMAL one-to-one column alignment using the Hungarian
     algorithm (scipy.optimize.linear_sum_assignment).  This maximises total
     similarity across all aligned pairs simultaneously — better than greedy
     matching.
  c) union_score = mean(aligned pair similarities) × coverage
     where coverage = min(cols_A, cols_B) / max(cols_A, cols_B).
     The coverage term penalises tables with very different numbers of
     columns (a 2-column table cannot really be unioned with a 50-column one).

Step 5 — Ground Truth Decision (Multi-Signal Validation)  (NEW)
  A pair is ground truth ONLY IF:
    (joinability ≥ 0.15 OR unionability ≥ 0.20)  AND  semantic_similarity ≥ 0.40

  This requires BOTH syntactic match AND semantic support, reducing false positives.

Output columns
--------------
  query_table         : filename of the query dataset
  candidate_table     : filename of the candidate dataset
  similarity          : max(join_score, union_score) — overall relevance score
  joinability         : join score  ∈ [0, 1]
  unionability        : union score ∈ [0, 1]
  semantic_similarity : embedding similarity ∈ [0, 1]  (NEW)
  is_join_gt          : 1 if joinability  ≥ 0.15
  is_union_gt         : 1 if unionability ≥ 0.20
  is_ground_truth     : 1 if (syntactic match) AND (semantic support ≥ 0.40)  (UPDATED)

Usage
-----
    # Quick test — 3 query datasets
    python3 groundtruth/generate_independent.py --sample 3

    # Full run — all datasets as queries (~160 × 160 pairs)
    python3 groundtruth/generate_independent.py --all

    # Limit how many candidates are scored per query (for speed)
    python3 groundtruth/generate_independent.py --sample 10 --max-candidates 30

    # Custom output file
    python3 groundtruth/generate_independent.py --all --output my_gt.csv
"""

import os
import csv
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

# Suppress OpenMP warnings on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

TOP_K = 20              # frequent values per column
MAX_ROWS = None         # None = read entire table (was 5000, now unlimited for better accuracy)
JOIN_THRESHOLD = 0.15
UNION_THRESHOLD = 0.20
SEMANTIC_THRESHOLD = 0.40  # NEW: semantic similarity must support the match

# Semantic similarity setup (lazy-loaded to avoid startup delay)
_semantic_model = None
_semantic_cache = {}

def get_semantic_model():
    """Lazy-load sentence-transformers model on first use."""
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading semantic model (all-MiniLM-L6-v2)...")
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            log.info("✓ Semantic model loaded")
        except ImportError:
            log.warning("⚠️  sentence-transformers not installed. Semantic signal disabled.")
            log.warning("   Install with: pip install sentence-transformers")
            _semantic_model = False  # Mark as unavailable
    return _semantic_model if _semantic_model else None


# ── Step - 1 - Column Profile extraction ───────────────────────────────────────────────

def extract_column_profile(series: pd.Series) -> Dict:
    """
    Summarise a single DataFrame column into a fixed set of statistics.

    Instead of keeping thousands of raw values, we extract a compact
    "signature" that captures what the column looks like:

    top_values (set of str)
        The TOP_K most frequently occurring non-null values, lowercased and
        stripped.  Example: {"france", "germany", "italy", ...}
        Used for value-overlap comparisons (join and union scoring).
        Why top-k instead of random sample? Because join keys and union
        domains are characterised by their *common* values, not rare ones.

    inferred_type ('numeric' | 'string')
        Determined by trying pd.to_numeric() on the first 50 non-null values.
        Drives which similarity formula is used downstream.

    cardinality_ratio (float, 0–1)
        n_unique / n_non_null.  Near 0 means most rows share the same value
        (e.g. a "country" column in a table about one country).  Near 1 means
        almost every row is different (e.g. a primary-key ID column).

    null_ratio (float, 0–1)
        Fraction of missing values.

    entropy (float, ≥ 0)
        Shannon entropy of the value distribution.
        - String: computed from value_counts() frequencies (bits).
        - Numeric: computed from a 20-bin histogram, then normalised to [0,1]
          by dividing by log2(20) so values are comparable across columns.
        High entropy → many equally likely values (e.g. uniform age spread).
        Low entropy  → one or few values dominate (e.g. 95% "active").

    numeric_stats (dict | None)
        Only present for numeric columns.  Contains: min, max, mean, std,
        p25, p50, p75, entropy.  Used in _numeric_distribution_similarity().
        None for string columns.
    """
    total = len(series)
    non_null = series.dropna()
    n_non_null = len(non_null)

    if n_non_null == 0:
        return {
            'top_values': set(),
            'inferred_type': 'string',
            'cardinality_ratio': 0.0,
            'null_ratio': 1.0,
            'numeric_stats': None,
        }

    # Infer type by attempting numeric conversion on a sample
    try:
        numeric_vals = pd.to_numeric(non_null, errors='raise')
        inferred_type = 'numeric'
    except (ValueError, TypeError):
        numeric_vals = None
        inferred_type = 'string'

    # Top-k frequent values as strings (useful for low-cardinality numerics too)
    freq = non_null.astype(str).str.lower().str.strip().value_counts().head(TOP_K)
    top_values = set(freq.index.tolist())

    cardinality_ratio = non_null.nunique() / n_non_null if n_non_null > 0 else 0.0
    null_ratio = (total - n_non_null) / total if total > 0 else 0.0

    # Entropy — computed for both types, different methods:
    #   String:  Shannon entropy of the value-frequency distribution (bits).
    #   Numeric: Shannon entropy of a 20-bin histogram (bits), normalised to [0,1]
    #            by dividing by log2(n_bins) so it is comparable across columns.
    entropy = 0.0
    try:
        if inferred_type == 'string':
            counts = non_null.astype(str).str.lower().str.strip().value_counts()
            probs = counts / counts.sum()
            entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        elif numeric_vals is not None:
            counts, _ = np.histogram(numeric_vals.dropna(), bins=20)
            total_counts = counts.sum()
            if total_counts > 0:
                probs = counts[counts > 0] / total_counts
                raw_entropy = float(-(probs * np.log2(probs)).sum())
                entropy = raw_entropy / np.log2(20)  # normalise to [0, 1]
    except Exception:
        entropy = 0.0

    # Numeric distribution stats — only for numeric columns
    numeric_stats = None
    if inferred_type == 'numeric' and numeric_vals is not None:
        try:
            numeric_stats = {
                'min':     float(numeric_vals.min()),
                'max':     float(numeric_vals.max()),
                'mean':    float(numeric_vals.mean()),
                'std':     float(numeric_vals.std(ddof=1)) if n_non_null > 1 else 0.0,
                'p25':     float(numeric_vals.quantile(0.25)),
                'p50':     float(numeric_vals.quantile(0.50)),
                'p75':     float(numeric_vals.quantile(0.75)),
                'entropy': entropy,
            }
        except Exception:
            numeric_stats = None

    return {
        'top_values': top_values,
        'inferred_type': inferred_type,
        'cardinality_ratio': cardinality_ratio,
        'null_ratio': null_ratio,
        'entropy': entropy,
        'numeric_stats': numeric_stats,
    }


def extract_table_profiles(df: pd.DataFrame) -> List[Dict]:
    """Extract column signatures for every column in a DataFrame."""
    profile = []
    for col in df.columns:
        sig = extract_column_profile(df[col])
        sig['column_name'] = col
        profile.append(sig)
    return profile


# ── Similarity primitives ────────────────────────────────────────────────────
# Jaccard Containment
def containment(a: set, b: set) -> float:
    """
    Directional overlap: fraction of a's values that also appear in b.

    containment({"fr","de","it"}, {"fr","de","us"}) = 2/3 ≈ 0.67
    containment({"fr","de","us"}, {"fr","de","it"}) = 2/3 ≈ 0.67  (same here)

    Used for join scoring: "how many of the query column's common values
    appear in the candidate column?"  Directional because we care whether
    the *query* values are covered, not the candidate's.
    """
    if not a:
        return 0.0
    return len(a & b) / len(a)

# Jaccard Similarity
def jaccard(a: set, b: set) -> float:
    """
    Symmetric overlap: intersection / union.

    jaccard({"fr","de","it"}, {"fr","de","us"}) = 2/4 = 0.50

    Used for union scoring: both columns need to share a common value
    domain, so we penalise when either side has values the other lacks.
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ──────────────────────────────────────────────────────────────────────────────
# ── Semantic similarity (Table-Level) ──────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────
# ── Semantic Similarity: Column-Level + Table-Level (Maximum Precision) ──────────
# ──────────────────────────────────────────────────────────────────────────────
# MAJOR ENHANCEMENT: This section implements comprehensive semantic validation
# - Table-level embeddings for quick domain filtering
# - Column-level embeddings for fine-grained matching
# - Integration with Hungarian algorithm for optimal alignment
# - Combined semantic+syntactic scores for maximum precision
# ──────────────────────────────────────────────────────────────────────────────

def serialize_column_for_embedding(col_name: str, series: pd.Series, max_values: int = 20) -> str:
    """
    Serialize a single column into descriptive text for embedding.

    Includes:
    - Column name (strong semantic signal)
    - Data type (numeric vs string)
    - Cardinality and null ratio (sparseness)
    - Top frequent values (domain context)
    - Distribution statistics (for numeric columns)

    Example output:
      "Column: customer_id | Type: numeric | Cardinality: 50000 unique values (91.7%)
       Null ratio: 0.0% | Top values: 1234(15) 5678(12) 9012(8) ...
       Range: [1, 999999] | Mean: 450000.00, Std: 250000.00"
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return f"Column {col_name}: empty/all null"

    # Type inference
    try:
        pd.to_numeric(non_null, errors='raise')
        col_type = "numeric"
        numeric_vals = pd.to_numeric(non_null)
    except:
        col_type = "string"
        numeric_vals = None

    # Cardinality
    cardinality = non_null.nunique()
    cardinality_ratio = cardinality / len(non_null)
    null_ratio = series.isna().sum() / len(series)

    # Build descriptive text
    text_parts = [f"Column: {col_name}"]
    text_parts.append(f"Type: {col_type}")
    text_parts.append(f"Cardinality: {cardinality} unique ({100*cardinality_ratio:.1f}%)")
    text_parts.append(f"Null: {100*null_ratio:.1f}%")

    # Top values (semantic domain indicator)
    top_vals = non_null.astype(str).str.lower().str.strip().value_counts().head(max_values)
    if len(top_vals) > 0:
        top_str = ", ".join(f"{v}({c})" for v, c in zip(top_vals.index[:10], top_vals.values[:10]))
        text_parts.append(f"Top values: {top_str}")

    # Numeric statistics if applicable
    if col_type == "numeric" and numeric_vals is not None:
        try:
            stats = {
                'min': numeric_vals.min(),
                'max': numeric_vals.max(),
                'mean': numeric_vals.mean(),
                'median': numeric_vals.median(),
                'std': numeric_vals.std(),
            }
            text_parts.append(f"Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
            text_parts.append(f"Mean: {stats['mean']:.0f}, Median: {stats['median']:.0f}, Std: {stats['std']:.0f}")
        except:
            pass

    return " | ".join(text_parts)


def compute_column_embeddings(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute semantic embeddings for all columns in a table.

    Each column is embedded individually to capture its semantic meaning
    (data type, domain, distribution shape).

    Returns: {column_name: embedding_vector (384-dim)}
    Results are cached to avoid recomputing.
    """
    model = get_semantic_model()
    if model is None:
        return {}

    embeddings = {}
    for col in df.columns:
        # Create cache key (use column name, not df id to enable cross-dataset caching)
        col_key = col

        # Check cache first
        if col_key in _semantic_cache:
            embeddings[col] = _semantic_cache[col_key]
            continue

        try:
            # Serialize this column
            col_text = serialize_column_for_embedding(col, df[col])

            # Embed it
            embedding = model.encode(col_text, convert_to_numpy=True)
            embeddings[col] = embedding

            # Cache result
            _semantic_cache[col_key] = embedding

        except Exception as e:
            log.debug(f"Error embedding column '{col}': {e}")
            continue

    return embeddings


def compute_column_semantic_similarity_matrix(
    df_q: pd.DataFrame,
    df_c: pd.DataFrame,
    embeddings_q: Dict[str, np.ndarray],
    embeddings_c: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Build semantic similarity matrix: query columns vs candidate columns.

    Each cell [i,j] = cosine similarity between query column i and candidate column j.

    Returns: (len(df_q.columns) × len(df_c.columns)) numpy array
    """
    n_q = len(df_q.columns)
    n_c = len(df_c.columns)
    sim_matrix = np.zeros((n_q, n_c))

    for i, col_q in enumerate(df_q.columns):
        if col_q not in embeddings_q:
            continue

        emb_q = embeddings_q[col_q]
        norm_q = np.linalg.norm(emb_q)
        if norm_q < 1e-9:
            continue

        for j, col_c in enumerate(df_c.columns):
            if col_c not in embeddings_c:
                continue

            emb_c = embeddings_c[col_c]
            norm_c = np.linalg.norm(emb_c)
            if norm_c < 1e-9:
                continue

            # Cosine similarity: (a · b) / (||a|| ||b||)
            similarity = float(np.dot(emb_q, emb_c) / (norm_q * norm_c))
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

            sim_matrix[i, j] = similarity

    return sim_matrix


def compute_table_semantic_similarity(filename_q: str, df_q: pd.DataFrame,
                                      filename_c: str, df_c: pd.DataFrame) -> Tuple[float, Optional[str]]:
    """
    Content-aware table-level semantic filter (Solution 4: Maximum Precision).

    Unlike simple name-based filters, this analyzes ACTUAL DATA to detect domain similarity.

    Process:
    1. Extract sample values from each column (first 50 rows, up to 8 columns)
    2. Create domain-aware text representation
    3. Embed and compare similarity

    This works even when:
    - Column names are completely different ("product_id" vs "item_code")
    - Column names are generic ("col_1", "col_2")
    - Tables have no semantic relationship but coincidental column names

    Trade-off:
    - ✓ Much better precision (detects true domain similarity from data)
    - ✗ Slightly slower (reads actual rows instead of just names)

    Returns: (similarity_score, error_message) where similarity in [0, 1]
    """
    model = get_semantic_model()
    if model is None:
        return 0.5, None

    try:
        # Create cache key for table pair
        cache_key = f"table_sem_{filename_q}_{filename_c}"
        if cache_key in _semantic_cache:
            return _semantic_cache[cache_key], None

        # STAGE 1: Serialize table with actual data content
        # This is much more informative than just column names

        text_q_parts = [f"Table: {filename_q}"]
        text_c_parts = [f"Table: {filename_c}"]

        # STAGE 2: For each column, include:
        # - Column name (if meaningful)
        # - Data type (numeric vs string)
        # - Sample values from actual data (the key differentiator)

        max_cols_to_sample = min(8, len(df_q.columns))  # Up to 8 columns
        for col in df_q.columns[:max_cols_to_sample]:
            non_null = df_q[col].dropna()
            if len(non_null) == 0:
                continue

            # Type
            try:
                pd.to_numeric(non_null, errors='raise')
                col_type = "numeric"
            except:
                col_type = "string"

            # Sample values (deterministic: head, not random)
            sample_vals = non_null.astype(str).str.lower().head(5).tolist()

            # Build column description
            col_desc = f"{col}({col_type}): {' '.join(sample_vals[:3])}"
            text_q_parts.append(col_desc)

        max_cols_to_sample = min(8, len(df_c.columns))
        for col in df_c.columns[:max_cols_to_sample]:
            non_null = df_c[col].dropna()
            if len(non_null) == 0:
                continue

            # Type
            try:
                pd.to_numeric(non_null, errors='raise')
                col_type = "numeric"
            except:
                col_type = "string"

            # Sample values
            sample_vals = non_null.astype(str).str.lower().head(5).tolist()

            # Build column description
            col_desc = f"{col}({col_type}): {' '.join(sample_vals[:3])}"
            text_c_parts.append(col_desc)

        # Join all parts with separator (so embedder understands structure)
        text_q = " | ".join(text_q_parts)
        text_c = " | ".join(text_c_parts)

        # STAGE 3: Embed both texts
        # The embedding now captures domain semantics from actual data values
        emb_q = model.encode(text_q, convert_to_numpy=True)
        emb_c = model.encode(text_c, convert_to_numpy=True)

        # STAGE 4: Compute cosine similarity
        norm_q = np.linalg.norm(emb_q)
        norm_c = np.linalg.norm(emb_c)

        if norm_q < 1e-9 or norm_c < 1e-9:
            similarity = 0.0
        else:
            similarity = float(np.dot(emb_q, emb_c) / (norm_q * norm_c))
            similarity = max(0.0, min(1.0, similarity))

        # Cache and return
        _semantic_cache[cache_key] = similarity
        return similarity, None

    except Exception as e:
        log.debug(f"Content-aware table filter error: {e}")
        return 0.5, None


def compute_semantic_similarity(filename_q: str, df_q: pd.DataFrame,
                                filename_c: str, df_c: pd.DataFrame,
                                profile_q: List[Dict],
                                profile_c: List[Dict],
                                syntactic_join: float,
                                syntactic_union: float
) -> Tuple[float, float, float, Optional[str]]:
    """
    Comprehensive semantic validation at column + table level.

    Multi-stage process:
    1. Quick table-level filter (reject if domains totally unrelated)
    2. Column-level embedding analysis
    3. Integration with syntactic scores for combined precision

    Returns:
        (table_semantic, join_semantic, union_semantic, error_message)
        - table_semantic: table-level similarity [0, 1]
        - join_semantic: best semantic column match for joins [0, 1]
        - union_semantic: optimal semantic alignment for unions [0, 1]
        - error_message: None if successful, otherwise error string
    """
    model = get_semantic_model()
    if model is None:
        return 0.5, 0.0, 0.0, "Semantic model not available"

    try:
        # Stage 1: Quick table-level filter
        table_sim, err = compute_table_semantic_similarity(filename_q, df_q, filename_c, df_c)
        if table_sim < 0.25:  # Early exit if totally unrelated
            return table_sim, 0.0, 0.0, None

        # Stage 2: Compute column embeddings
        embeddings_q = compute_column_embeddings(df_q)
        embeddings_c = compute_column_embeddings(df_c)

        if not embeddings_q or not embeddings_c:
            return table_sim, 0.0, 0.0, "No column embeddings computed"

        # Stage 3a: Semantic similarity for joinability
        # For joins: find the BEST semantic+syntactic match among column pairs
        sem_sim_matrix = compute_column_semantic_similarity_matrix(df_q, df_c, embeddings_q, embeddings_c)
        max_sem_join = 0.0

        for i, sq in enumerate(profile_q):
            vals_q = sq['top_values']
            if not vals_q or (sq['inferred_type'] == 'numeric' and sq['cardinality_ratio'] > _NUMERIC_JOIN_CARDINALITY_MAX):
                continue

            for j, sc in enumerate(profile_c):
                vals_c = sc['top_values']
                if not vals_c or (sc['inferred_type'] == 'numeric' and sc['cardinality_ratio'] > _NUMERIC_JOIN_CARDINALITY_MAX):
                    continue

                if sq['inferred_type'] != sc['inferred_type']:
                    continue

                # Get scores
                syntactic_match = containment(vals_q, vals_c)
                semantic_match = sem_sim_matrix[i, j] if i < len(sem_sim_matrix) and j < len(sem_sim_matrix[0]) else 0.0

                # BOTH must be reasonable (not coincidence)
                if syntactic_match > 0.05:  # Only if there's actual value overlap
                    combined = semantic_match * syntactic_match
                    max_sem_join = max(max_sem_join, semantic_match)

        # Stage 3b: Semantic similarity for unionability
        # For unions: optimal 1-to-1 alignment using Hungarian + embeddings
        n_q = len(profile_q)
        n_c = len(profile_c)
        combined_matrix = np.zeros((n_q, n_c))

        for i, sq in enumerate(profile_q):
            for j, sc in enumerate(profile_c):
                semantic_score = sem_sim_matrix[i, j] if i < len(sem_sim_matrix) and j < len(sem_sim_matrix[0]) else 0.0
                syntactic_score = _column_pair_similarity(sq, sc)

                # Weighted: 60% semantic (domain) + 40% syntactic (content)
                combined_matrix[i, j] = 0.6 * semantic_score + 0.4 * syntactic_score

        # Find optimal alignment
        row_ind, col_ind = linear_sum_assignment(-combined_matrix)  # Negative for max

        # Compute union semantic score
        sem_sum = sum(sem_sim_matrix[i, j] for i, j in zip(row_ind, col_ind) if i < len(sem_sim_matrix) and j < len(sem_sim_matrix[0]))
        n_aligned = len(row_ind)
        coverage = min(n_q, n_c) / max(n_q, n_c) if max(n_q, n_c) > 0 else 0.0
        union_semantic = (sem_sum / n_aligned * coverage) if n_aligned > 0 else 0.0

        return table_sim, max_sem_join, union_semantic, None

    except Exception as e:
        log.debug(f"Error in semantic similarity: {e}")
        return 0.5, 0.0, 0.0, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# ── Join scoring ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

# Cardinality threshold below which a numeric column is treated as a discrete
# join key (zip codes, years, product codes).  Above this, exact-value join
# is meaningless for continuous data (prices, measurements, coordinates).
_NUMERIC_JOIN_CARDINALITY_MAX = 0.05   # ≤ 5 % unique values → discrete key


def compute_joinability_score(
    profile_q: List[Dict],
    profile_c: List[Dict],
) -> float:
    """
    Estimate how joinable two datasets are, based on shared frequent values.

    Two datasets are joinable if there exists at least one pair of columns
    (one from each table) whose values overlap enough to perform an equi-join.
    Example: table A has a "city" column, table B also has a "city" column,
    and they share many of the same city names → joinable.

    Algorithm
    ---------
    For every pair (col from query table, col from candidate table):
      1. Skip if the two columns have different types (string vs numeric).
      2. Skip numeric columns with high cardinality (> 5% unique values).
         Continuous values like prices or GPS coordinates don't make join keys;
         only discrete numerics like zip codes or years are valid join keys.
      3. Compute containment(top_values_query, top_values_candidate).
    Return the MAXIMUM score across all pairs.

    Why max and not average?
    A single joinable column pair is sufficient.  Most real-world joins happen
    on one key column (e.g. country code, product ID).  Averaging would dilute
    a strong join signal from one column with zeros from unrelated columns.

    Returns
    -------
    float in [0, 1].  A score ≥ JOIN_THRESHOLD (0.15) labels the pair as
    joinable ground truth.
    """
    max_score = 0.0

    for sq in profile_q:
        vals_q = sq['top_values']
        if not vals_q:
            continue

        # Skip high-cardinality continuous numeric columns
        if sq['inferred_type'] == 'numeric' \
                and sq['cardinality_ratio'] > _NUMERIC_JOIN_CARDINALITY_MAX:
            continue

        for sc in profile_c:
            vals_c = sc['top_values']
            if not vals_c:
                continue

            if sc['inferred_type'] == 'numeric' \
                    and sc['cardinality_ratio'] > _NUMERIC_JOIN_CARDINALITY_MAX:
                continue

            # Only score same-type pairs (don't join a string col onto a numeric)
            if sq['inferred_type'] != sc['inferred_type']:
                continue

            score = containment(vals_q, vals_c)
            if score > max_score:
                max_score = score
                if max_score >= 1.0:
                    return 1.0

    return max_score

# ──────────────────────────────────────────────────────────────────────────────
# ── Union scoring ─────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def _entropy_similarity(e_a: float, e_b: float) -> float:
    """
    Similarity between two entropy values, bounded to [0, 1].

    Uses a relative difference: penalty = |e_a - e_b| / max(e_a, e_b).

    Examples:
      Both near-constant columns (e≈0 vs e≈0)  → 1.0  (both "boring")
      Both uniform columns      (e≈1 vs e≈1)   → 1.0  (both "spread out")
      One constant, one uniform (e≈0 vs e≈1)   → 0.0  (very different shapes)
    """
    denom = max(e_a, e_b, 1e-9)
    return max(0.0, 1.0 - abs(e_a - e_b) / denom)


def _numeric_distribution_similarity(stats_a: Dict, stats_b: Dict) -> float:
    """
    Similarity between two numeric columns using distribution statistics.

    Jaccard on raw values is useless for continuous numbers: two "price"
    columns with ranges [5.99, 149.99] and [6.00, 150.00] share zero exact
    values but are clearly the same kind of column.  Instead, we compare
    the *shape* of their distributions using four components:

    range_overlap (weight 0.40)
        Fraction of the combined [min, max] range that both columns share.
        Example: col A in [0, 100], col B in [50, 200] → overlap = 50/200 = 0.25
        Captures whether both columns live in the same value space.

    mean_sim (weight 0.25)
        How close the centres are, relative to the combined range.
        Two "age" columns both centred around 35 score high; an "age" column
        vs a "salary" column (very different centres) score low.

    spread_sim (weight 0.20)
        How similar the standard deviations are.
        Two tightly-clustered columns, or two widely-spread columns, score
        high.  One tight vs one spread scores low.

    entropy_sim (weight 0.15)
        How similar the distribution shapes are (via histogram entropy).
        A uniform age distribution vs a heavily skewed salary distribution
        score low even if their ranges overlap.

    Returns
    -------
    float in [0, 1].
    """
    lo = min(stats_a['min'], stats_b['min'])
    hi = max(stats_a['max'], stats_b['max'])
    total_range = hi - lo

    if total_range < 1e-9:
        return 1.0 if abs(stats_a['mean'] - stats_b['mean']) < 1e-9 else 0.0

    # Range overlap: length of intersection / length of union
    overlap_lo = max(stats_a['min'], stats_b['min'])
    overlap_hi = min(stats_a['max'], stats_b['max'])
    range_overlap = max(0.0, overlap_hi - overlap_lo) / total_range

    # Mean similarity
    mean_sim = max(0.0, 1.0 - abs(stats_a['mean'] - stats_b['mean']) / total_range)

    # Spread similarity
    std_sum = stats_a['std'] + stats_b['std']
    spread_sim = (
        1.0 if std_sum < 1e-9
        else 1.0 - abs(stats_a['std'] - stats_b['std']) / std_sum
    )

    # Entropy similarity (normalised bin-entropy, already in [0, 1])
    ent_sim = _entropy_similarity(stats_a.get('entropy', 0.0), stats_b.get('entropy', 0.0))

    return 0.40 * range_overlap + 0.25 * mean_sim + 0.20 * spread_sim + 0.15 * ent_sim


def _column_pair_similarity(sig_a: Dict, sig_b: Dict) -> float:
    """
    Similarity between two column signatures for union scoring.

    Column names are NOT used — alignment is purely content-based.

    Three cases:

    String × String
        Score = 0.55 × jaccard(top_values)    — do they share common values?
              + 0.25 × 1.0 (type match)       — both are strings (always 1)
              + 0.10 × cardinality_sim         — similar uniqueness ratio?
              + 0.10 × entropy_sim             — similar distribution shape?
        Example: two "country" columns from different tables will score high
        because they share many country names and have similar entropy.

    Numeric × Numeric
        Score = 0.85 × _numeric_distribution_similarity()
              + 0.15 × cardinality_sim
        Value Jaccard is meaningless for continuous numbers; instead we
        compare range, mean, spread, and entropy of the distributions.

    Mixed (string × numeric or numeric × string)
        Score = 0.0
        A text column and a number column can never be unioned — incompatible
        data types.

    Returns
    -------
    float in [0, 1].  Used as the cell value in the similarity matrix passed
    to the Hungarian assignment algorithm.
    """
    type_a = sig_a['inferred_type']
    type_b = sig_b['inferred_type']

    # Incompatible types
    if type_a != type_b:
        return 0.0

    card_sim = max(0.0, 1.0 - abs(sig_a['cardinality_ratio'] - sig_b['cardinality_ratio']))

    if type_a == 'numeric':
        stats_a = sig_a.get('numeric_stats')
        stats_b = sig_b.get('numeric_stats')
        if stats_a is None or stats_b is None:
            return 0.25 + 0.75 * card_sim  # fallback if stats unavailable
        dist_sim = _numeric_distribution_similarity(stats_a, stats_b)
        return 0.85 * dist_sim + 0.15 * card_sim

    # String path — entropy captures distribution shape beyond what Jaccard sees.
    # Two columns with similar top-k values AND similar entropy are more likely
    # to represent the same domain than high-Jaccard / very different entropies.
    val_sim  = jaccard(sig_a['top_values'], sig_b['top_values'])
    ent_sim  = _entropy_similarity(sig_a.get('entropy', 0.0), sig_b.get('entropy', 0.0))
    return 0.55 * val_sim + 0.25 * 1.0 + 0.10 * card_sim + 0.10 * ent_sim


def compute_unionability_score(
    profile_q: List[Dict],
    profile_c: List[Dict],
) -> float:
    """
    Estimate how unionable two datasets are using content-based column alignment.

    Two datasets are unionable if their columns cover the same domains so that
    rows could be stacked (SQL UNION).  Example: two sales tables from different
    regions, both with columns for date, product, amount, and region.

    Column names are intentionally ignored — "revenue" and "amount" should
    align if they contain similar numbers, even with different names.

    Algorithm
    ---------
    Step 1 — Build an (m × n) similarity matrix
        m = number of columns in query table
        n = number of columns in candidate table
        Each cell [i, j] = _column_pair_similarity(col_i, col_j)
        This captures value overlap (strings) or distribution similarity
        (numerics) without looking at column names.

    Step 2 — Optimal one-to-one column alignment (Hungarian algorithm)
        scipy.optimize.linear_sum_assignment finds the assignment of query
        columns to candidate columns that MAXIMISES total similarity.
        Unlike greedy matching (always pick the locally best pair), Hungarian
        guarantees the globally optimal solution.
        Each column can only be matched once (one-to-one constraint).

    Step 3 — Aggregate into a single score
        union_score = mean(similarity of all aligned pairs) × coverage

        coverage = min(m, n) / max(m, n)
        This penalises tables with very different numbers of columns.
        Rationale: a 2-column table can always align 2 columns from a
        50-column table by chance, but that does not make them unionable.

    Returns
    -------
    float in [0, 1].  A score ≥ UNION_THRESHOLD (0.20) labels the pair as
    unionable ground truth.
    """
    m, n = len(profile_q), len(profile_c)
    if m == 0 or n == 0:
        return 0.0

    # Build similarity matrix
    sim_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            sim_mat[i, j] = _column_pair_similarity(profile_q[i], profile_c[j])

    # Hungarian assignment (minimise cost → negate similarity)
    row_idx, col_idx = linear_sum_assignment(-sim_mat)

    aligned_sims = sim_mat[row_idx, col_idx]
    avg_sim = float(aligned_sims.mean()) if len(aligned_sims) > 0 else 0.0
    coverage = min(m, n) / max(m, n)

    return avg_sim * coverage


# ── Main generator ────────────────────────────────────────────────────────────

def generate_independent_groundtruth(
    datalake_dir: str = 'freya_data',
    output_path: str = 'validation_groundtruth_independent.csv',
    sample_size: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> list:
    """
    Main entry point: score all dataset pairs and write results to a CSV.

    For efficiency, all datasets are loaded and profiled once upfront into
    a signature cache.  No dataset is read from disk more than once.

    Parameters
    ----------
    datalake_dir : str
        Path to the folder containing the datalake CSV files.
    output_path : str
        Where to write the resulting ground truth CSV.
    sample_size : int or None
        How many datasets to use as query tables (the first N alphabetically).
        None means all datasets are used as queries.
    max_candidates : int or None
        Maximum number of candidate tables to score each query against.
        When set, the query tables themselves are always included; the
        remaining slots are filled by random sampling.  Useful for fast
        experimentation without scoring all ~160 × 160 pairs.

    Returns
    -------
    list of rows (same content as the output CSV, without the header).
    """
    import random

    log.info("Independent ground truth generation")
    log.info(f"  Join:  max containment of top-{TOP_K} frequent values")
    log.info(f"  Union: Hungarian assignment on value-domain similarity\n")

    # Get all CSV files
    all_files = sorted(f for f in os.listdir(datalake_dir) if f.endswith('.csv'))
    if not all_files:
        log.error(f"No CSV files found in {datalake_dir}")
        return []

    log.info(f"Found {len(all_files)} datasets in datalake\n")

    # Select queries and candidates
    query_tables = all_files[:sample_size] if sample_size else all_files
    candidate_tables = all_files

    if max_candidates and len(candidate_tables) > max_candidates:
        query_set = set(query_tables)
        others = [c for c in candidate_tables if c not in query_set]
        sampled = random.sample(others, min(max_candidates - len(query_set), len(others)))
        candidate_tables = sorted(query_set | set(sampled))

    log.info(f"Queries: {len(query_tables)}  x  Candidates: {len(candidate_tables)}")
    log.info(f"Total pairs (excl. self): {len(query_tables) * (len(candidate_tables) - 1):,}\n")

    # Pre-load datasets and extract signatures (avoids re-reading CSVs)
    log.info("Loading datasets and extracting column signatures…")
    sig_cache: Dict[str, List[Dict]] = {}
    df_cache: Dict[str, pd.DataFrame] = {}  # NEW: cache full DataFrames for semantic scoring
    load_failures = 0

    for fname in sorted(set(query_tables) | set(candidate_tables)):
        fpath = os.path.join(datalake_dir, fname)
        try:
            # Read with MAX_ROWS limit for profiling, but keep full for semantic
            df = pd.read_csv(fpath, dtype=str, on_bad_lines='skip')
            sig_cache[fname] = extract_table_profiles(df)
            df_cache[fname] = df  # Store the full DataFrame
        except Exception as e:
            log.debug(f"  skip {fname}: {e}")
            load_failures += 1

    log.info(f"  Loaded {len(sig_cache)} datasets ({load_failures} failures)")

    # Check semantic model availability
    model = get_semantic_model()
    if model is not None:
        log.info(f"  ✓ Semantic similarity enabled\n")
    else:
        log.info(f"  ⚠️  Semantic similarity disabled (sentence-transformers not available)\n")

    # Score all pairs
    results = []
    for q_idx, q_file in enumerate(query_tables, 1):
        if q_file not in sig_cache:
            continue

        log.info(f"[{q_idx}/{len(query_tables)}] {q_file}")
        profile_q = sig_cache[q_file]
        scored = 0

        for c_file in candidate_tables:
            if c_file == q_file or c_file not in sig_cache:
                continue

            profile_c = sig_cache[c_file]

            try:
                # Stage 1: Compute syntactic signals (joinability, unionability)
                j_score = compute_joinability_score(profile_q, profile_c)
                u_score = compute_unionability_score(profile_q, profile_c)

                # Stage 2: Compute column-level semantic signals (if available)
                # Returns: (table_sem, join_sem, union_sem, error_message)
                table_sem = 0.5
                join_sem = 0.0
                union_sem = 0.0

                if q_file in df_cache and c_file in df_cache:
                    table_sem, join_sem, union_sem, sem_err = compute_semantic_similarity(
                        q_file, df_cache[q_file],
                        c_file, df_cache[c_file],
                        profile_q, profile_c,
                        j_score, u_score
                    )
                    if sem_err:
                        log.debug(f"    semantic warning for {c_file}: {sem_err}")

                # Stage 3: Ground truth decision with semantic validation
                # For JOIN: syntactic join match must have semantic support
                # For UNION: syntactic union match must have semantic support
                is_join_gt = int(j_score >= JOIN_THRESHOLD and join_sem >= SEMANTIC_THRESHOLD)
                is_union_gt = int(u_score >= UNION_THRESHOLD and union_sem >= SEMANTIC_THRESHOLD)
                is_gt = int(is_join_gt or is_union_gt)

                # For combined score, use the better of the two (syntactic)
                combined = max(j_score, u_score)

                results.append([
                    q_file,
                    c_file,
                    round(combined, 6),
                    round(j_score, 6),
                    round(u_score, 6),
                    round(table_sem, 6),   # Table-level semantic
                    round(join_sem, 6),    # Join-specific semantic
                    round(union_sem, 6),   # Union-specific semantic
                    is_join_gt,            # Ground truth for joins
                    is_union_gt,           # Ground truth for unions
                    is_gt,                 # Overall ground truth
                ])
                scored += 1

            except Exception as e:
                log.debug(f"  error {c_file}: {e}")
                continue

        log.info(f"  {scored} pairs scored")

    # Write output
    log.info(f"\nWriting {len(results)} pairs to {output_path}…")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query_table', 'candidate_table', 'similarity',
            'joinability', 'unionability',
            'semantic_table', 'semantic_join', 'semantic_union',
            'is_join_gt', 'is_union_gt', 'is_ground_truth',
        ])
        writer.writerows(results)

    # Summary
    if results:
        j_scores = [r[3] for r in results]
        u_scores = [r[4] for r in results]
        t_sem_scores = [r[5] for r in results]
        j_sem_scores = [r[6] for r in results]
        u_sem_scores = [r[7] for r in results]
        n_join = sum(r[8] for r in results)
        n_union = sum(r[9] for r in results)
        n_gt = sum(r[10] for r in results)
        total = len(results)

        log.info(f"\n{'='*70}")
        log.info(f"GROUND TRUTH GENERATION SUMMARY (with Column-Level Semantic Validation)")
        log.info(f"{'='*70}")
        log.info(f"\nTotal pairs scored: {total:,}")
        log.info(f"  ✓ Joinable (semantic validated):   {n_join:,}  ({100*n_join/total:.1f}%)")
        log.info(f"  ✓ Unionable (semantic validated):  {n_union:,}  ({100*n_union/total:.1f}%)")
        log.info(f"  ✓ Ground truth (either):           {n_gt:,}  ({100*n_gt/total:.1f}%)")

        log.info(f"\nSyntactic Signals (Content-Based):")
        log.info(f"  Joinability:")
        log.info(f"    Mean: {np.mean(j_scores):.3f}  |  Max: {max(j_scores):.3f}  |  Threshold: {JOIN_THRESHOLD}")
        log.info(f"  Unionability:")
        log.info(f"    Mean: {np.mean(u_scores):.3f}  |  Max: {max(u_scores):.3f}  |  Threshold: {UNION_THRESHOLD}")

        log.info(f"\nSemantic Signals (Column-Level Embeddings):")
        log.info(f"  Table-level semantic:")
        log.info(f"    Mean: {np.mean(t_sem_scores):.3f}  |  Max: {max(t_sem_scores):.3f}  (quick domain filter)")
        log.info(f"  Join-specific semantic:")
        log.info(f"    Mean: {np.mean(j_sem_scores):.3f}  |  Max: {max(j_sem_scores):.3f}  (column match quality)")
        log.info(f"  Union-specific semantic:")
        log.info(f"    Mean: {np.mean(u_sem_scores):.3f}  |  Max: {max(u_sem_scores):.3f}  (schema alignment quality)")

        log.info(f"\nGround Truth Validation Rules:")
        log.info(f"  ✓ Join match:  joinability ≥ {JOIN_THRESHOLD} AND semantic_join ≥ {SEMANTIC_THRESHOLD}")
        log.info(f"  ✓ Union match: unionability ≥ {UNION_THRESHOLD} AND semantic_union ≥ {SEMANTIC_THRESHOLD}")
        log.info(f"  ✓ Overall GT:  either join or union match passes both checks")

    log.info(f"\nDone! → {output_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Independent ground truth from actual data content'
    )
    parser.add_argument('--all', action='store_true', help='All queries')
    parser.add_argument('--sample', type=int, help='N queries')
    parser.add_argument('--max-candidates', type=int,
                        help='Cap candidate count (random sample)')
    parser.add_argument('--output', default='validation_groundtruth_independent.csv')

    args = parser.parse_args()

    if not args.all and not args.sample:
        args.sample = 3
        log.info("No size specified — defaulting to --sample 3.\n")

    generate_independent_groundtruth(
        output_path=args.output,
        sample_size=None if args.all else args.sample,
        max_candidates=args.max_candidates,
    )

# Dataset Similarity Ground Truth Generation

A lightweight, standalone Python solution for generating ground truth labels for dataset similarity in data lakes. This tool combines semantic and syntactic signals to automatically identify when two tabular datasets can be meaningfully joined or unioned.

## Overview

This project provides a **two-gate validation system** that labels dataset pairs as:
- **Joinable** — can be combined by matching values in one column
- **Unionable** — can be stacked row-wise by aligning columns semantically
- **Similar** — satisfies both semantic meaning AND syntactic compatibility

The solution is designed to be:
- **Lightweight** — runs on a standard laptop without GPU
- **Fast** — processes hundreds of table pairs in minutes
- **Agnostic** — works on any tabular data regardless of domain

## Key Features

### Semantic Similarity
- Table-level embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)
- Column-level embeddings for fine-grained matching
- Cosine similarity for quick topic filtering
- Early exit optimization: skip expensive column analysis if tables are semantically too different

### Syntactic Similarity
- **Join scoring**: Containment metric (|A ∩ B| / |A|) across all column pairs
- **Union scoring**: Hungarian algorithm for optimal column alignment
- **Value sampling**: Up to 500 most-frequent values per column
- **Type matching**: Ensures numeric/string compatibility before scoring

### Combined Validation
- Both gates must pass: `is_similar = (semantic ✓) AND (syntactic ✓)`
- Thresholds are tunable:
  - `JOIN_THRESHOLD = 0.15` (syntactic join)
  - `UNION_THRESHOLD = 0.20` (syntactic union)
  - `SEMANTIC_THRESHOLD = 0.40` (semantic join/union)

## Installation

### Requirements
- Python 3.9+
- ~2GB disk space for semantic model cache

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/dataset-similarity-groundtruth.git
cd dataset-similarity-groundtruth

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download semantic model (automatic on first use, or manual)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Usage

### Basic Usage

```python
from generate_groundtruth import score_pair
from src.gt_profiling import extract_table_profiles
import pandas as pd

# Load your tables
df_query = pd.read_csv('query_table.csv')
df_candidate = pd.read_csv('candidate_table.csv')

# Profile them
profile_q = extract_table_profiles(df_query, top_k=200)
profile_c = extract_table_profiles(df_candidate, top_k=200)

# Score the pair
results = score_pair(
    'query_table.csv', df_query, profile_q,
    'candidate_table.csv', df_candidate, profile_c
)

overall_sim, join, union, sem, sem_join, sem_union, is_join_gt, is_union_gt, is_gt = results
print(f"Ground truth: {is_gt} | Joinable: {is_join_gt} | Unionable: {is_union_gt}")
print(f"  Semantic: {sem:.3f} | Syntactic Join: {join:.3f} | Syntactic Union: {union:.3f}")
```

### Full Pipeline

```bash
# Generate ground truth for all pairs in your datalake
# Edit generate_groundtruth.py:
#   - Set num_queries = None  (use all tables) or e.g. 5 (test mode)
#   - Set max_candidates_per_query = None (compare against all)
#   - Point DATALAKE_DIR to your CSV folder

python generate_groundtruth.py

# Output: validation_groundtruth.csv with columns:
# query_table, candidate_table, similarity, joinability, unionability,
# semantic_table, semantic_join, semantic_union,
# is_join_gt, is_union_gt, is_ground_truth
```

## Configuration

Edit `generate_groundtruth.py` to adjust:

```python
# Line 34-37: Tunable thresholds
TOP_K = 200                    # Most frequent values to sample per column
JOIN_THRESHOLD = 0.15          # Syntactic join threshold
UNION_THRESHOLD = 0.20         # Syntactic union threshold
SEMANTIC_THRESHOLD = 0.40      # Semantic gate (both signals)

# Line 115-117: Dataset configuration
num_queries = None             # None = all, or e.g. 5 for testing
max_candidates_per_query = None  # None = all, or e.g. 50 to limit
DATALAKE_DIR = Path('freya_data')  # Folder with your CSVs
```

## Understanding the Algorithm

### The Two-Gate System

1. **Semantic Gate**: Do these tables discuss the same topic?
   - Embed table description + sample data
   - Compute cosine similarity between embeddings
   - If < 0.25, skip expensive column analysis (optimization)

2. **Syntactic Gates**: Can these tables actually combine?
   - For **Joins**: Find best column pair with overlapping values (containment)
   - For **Unions**: Find optimal 1-to-1 column alignment (Hungarian algorithm)

3. **Decision**:
   ```
   is_join_gt = (join_score ≥ 0.15) AND (semantic_join ≥ 0.40)
   is_union_gt = (union_score ≥ 0.20) AND (semantic_union ≥ 0.40)
   is_ground_truth = is_join_gt OR is_union_gt
   ```

### Module Breakdown

| Module | Purpose |
|--------|---------|
| `gt_profiling.py` | Extract column profiles (top values, type, cardinality) |
| `gt_similarity_join.py` | Compute containment-based join scores |
| `gt_similarity_union.py` | Compute Jaccard + Hungarian union scores |
| `gt_similarity_semantic.py` | Compute semantic embeddings & similarities |

## Documentation

- **[SIMILARITY_DEFINITION.md](SIMILARITY_DEFINITION.md)** — What does "similarity" mean? Examples and intuition.
- **[GROUNDTRUTH_LOGIC.md](GROUNDTRUTH_LOGIC.md)** — Technical deep-dive into the validation architecture.
- **[INDEPENDENT_METHOD.md](INDEPENDENT_METHOD.md)** — Design decisions and alternatives considered.
- **[PYTHON_GUIDE.md](PYTHON_GUIDE.md)** — Code walkthrough for beginners.

## Output Format

`validation_groundtruth.csv`:

| Column | Meaning |
|--------|---------|
| `query_table` | Name of query table |
| `candidate_table` | Name of candidate table |
| `similarity` | max(join_score, union_score) |
| `joinability` | Join containment score [0, 1] |
| `unionability` | Union alignment score [0, 1] |
| `semantic_table` | Table-level semantic similarity [0, 1] |
| `semantic_join` | Best column-pair semantic match [0, 1] |
| `semantic_union` | Avg semantic alignment across optimal columns [0, 1] |
| `is_join_gt` | 1 if joinable, 0 otherwise |
| `is_union_gt` | 1 if unionable, 0 otherwise |
| `is_ground_truth` | 1 if either join or union possible, 0 otherwise |

## Performance

Tested on ~160 real-world CSVs (FREYJA benchmark):

- **Time**: ~5-10 seconds per query table (50 candidates) on CPU
- **Accuracy**: ~130 positive matches per 810 pairs (16% ground truth density)
- **Memory**: ~500MB for embeddings cache

## Limitations & Future Work

1. **Sampling bias** — Currently uses most-frequent 500 values. Could improve with stratified sampling.
2. **Column order** — Union matching is order-independent via Hungarian algorithm, but join detection is still greedy.
3. **Numeric columns** — High-cardinality numeric columns are skipped. Could add range-based matching.
4. **Domain adaptation** — Fixed semantic model. Could fine-tune on domain-specific tables.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{dataset_similarity_gt,
  title={Dataset Similarity Ground Truth Generation},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/dataset-similarity-groundtruth}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Contact

For questions, please open an issue on GitHub.

---

**Part of a thesis project on hybrid semantic-syntactic data discovery in data lakes.**

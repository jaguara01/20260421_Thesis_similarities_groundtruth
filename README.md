# Dataset similarties - Groundtruth labels

To implement the lightweigth solution for dataset similarties, we first need to find a ground truth to validate our solution.
Because I couldn't find any documentation that could validate my work. I decided to implement a more simple approach but heavy.

This groundtruth solution is based on the following definition of similarity:

Two datasets are similar:

1. If their semantic is similar at the table level (Based on 500 rows)
2. **AND** if they can be joined or unioned together (syntactic + semantic at column level)

Based on this definition, I build a groundtruth comparing all csv from Freya. The entire process took me 1h30 (Macbook Pro M3 - 36GB RAM)

## Workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Two Tables to Compare                       │
│                    (Query vs Candidate)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
   ┌─────────────┐                    ┌──────────────┐
   │   PROFILE   │                    │   PROFILE    │
   │   TABLES    │                    │   TABLES     │
   └──────┬──────┘                    └───────┬──────┘
          │                                   │
          └───────────────┬───────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
    ┌──────────────┐              ┌──────────────┐
    │   SEMANTIC   │              │  SYNTACTIC   │
    │  SIMILARITY  │              │  SIMILARITY  │
    │  (Embeddings)│              │  (Overlaps)  │
    └──────┬───────┘              └───────┬──────┘
           │                              │
    ┌──────▼──────┬──────────────────────▼──────┐
    │             │                             │
    ▼             ▼                             ▼
┌─────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐
│ Table   │ │ Column   │ │ Joinability  │ │ Unionability │
│ Semantic│ │ Semantic │ │   Score      │ │    Score     │
│ Score   │ │ Score    │ │              │ │              │
└────┬────┘ └────┬─────┘ └──────┬───────┘ └────────┬─────┘
     │           │              │                  │
     └───────────┴──────────────┴──────────────────┘
                        │
                        ▼
            ┌────────────────────────────┐
            │   COMBINED VALIDATION      │
            │  (Check Both Signals)      │
            └────────────────────────────┘
                        │
                        ▼
            ┌────────────────────────────┐
            │  Ground Truth Label        │
            │  (Join / Union / Neither)  │
            └────────────────────────────┘
```

---

## 1: Light profile of the tables

- **Top values** — the most common 200 unique values
- **Type** — numeric or string
- **Cardinality** — how many unique values exist (0 = all same, 1 = all different)
- **Null ratio** — how many missing values

## 2: Semantic similarity

### 2.1: Table-level semantic score

1. Serealization of the table to text
2. Embed this text into a 384-dimensional vector
3. Compare with another table's vector
4. Compute cosine similarity

This is used as a first filter

### 2.2 Column-level semantic Scores

1. Serealization of each column to text
2. Embed this text into a 384-dimensional vector
3. **Semantic Join Score** — best matching column pair
4. **Semantic Union Score** — average alignment of all columns

This allow us to avoid false positive when comparing 2 columns

---

## 3: Compute Join similarity

### Containment method

```
containment(A, B) = |A ∩ B| / |A|
+ cardinality similarity check
```

We are looking for the maximum value of containent score because only one good join is enough to find similarity.

## 4: Compute union similarity

### 4.1 Column pair similarity

For each pair of columns, we compute similarity based on:

- Value overlap (Jaccard) — 70% weight
- Cardinality similarity — 30% weight

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

### 4.2 Optimal union with the Hungarian algorithm

To find the best union possible, I'm using the Hungarian algorithm

### Union Score Calculation

```
Union Score = (Average alignment similarity) × Coverage penalty

Coverage = min(n_cols_query, n_cols_candidate) / max(n_cols_query, n_cols_candidate)

```

The coverage penalty is used to penalize large column count mismatches

---

## 5: Groundtruth score

We use both syntactic and semantic signals to avoid false positives.

#### Syntactic score

```
Syntactic Join:  join_score ≥ 0.15
Syntactic Union: union_score ≥ 0.20
```

#### Semantic score

```
Semantic Join:  semantic_join ≥ 0.40
Semantic Union: semantic_union ≥ 0.40
```

### Final Decision

```python
is_join_gt = 1 if (join_score >= JOIN_THRESHOLD and semantic_join >= SEMANTIC_THRESHOLD) else 0
is_union_gt = 1 if (union_score >= UNION_THRESHOLD and semantic_union >= SEMANTIC_THRESHOLD) else 0
is_ground_truth = 1 if (is_join_gt or is_union_gt) else 0
```

---

## Current thresholds

| Parameter            | Value | Meaning                                          |
| -------------------- | ----- | ------------------------------------------------ |
| `JOIN_THRESHOLD`     | 0.15  | Syntactic for joins                              |
| `UNION_THRESHOLD`    | 0.20  | Syntactic for unions                             |
| `SEMANTIC_THRESHOLD` | 0.40  | Semantic gate for both                           |
| `TOP_K`              | 200   | Top 200 frequent values per column for syntactic |

```
Syntactic thresholds (0.15, 0.20):
  - Loose: Allow for different column names
  - Example: customer_id vs cust_id might have 20% variance
  - But we catch errors with semantic gate

Semantic threshold (0.40):
  - Strict: Require strong semantic match
  - Example: "customer_id" and "country" won't both be 0.40+
  - Acts as final safety check
```

---

## Decision workflow

```
Input: Two tables (Query, Candidate)
       │
       ├─ Profile both tables
       │
       ├─ Compute semantic_table (quick filter)
       │  └─ If < 0.25 → too different, skip expensive analysis
       │
       ├─ Compute join_score (syntactic overlap)
       │
       ├─ Compute union_score (syntactic alignment)
       │
       ├─ Compute semantic_join (column meaning for join)
       │
       ├─ Compute semantic_union (column meaning for union)
       │
       └─ Decide:
          │
          ├─ If join_score ≥ 0.15 AND semantic_join ≥ 0.40
          │  → is_join_gt = 1 (joinable)
          │
          ├─ If union_score ≥ 0.20 AND semantic_union ≥ 0.40
          │  → is_union_gt = 1 (unionable)
          │
          └─ If either is 1 → is_ground_truth = 1
```

---

(The following was generated with AI)

## Installation

### Requirements

- Python 3.9+
- ~2GB disk space for semantic model cache

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/jaguara01/20260421_Thesis_similarities_groundtruth.git
cd 20260421_Thesis_similarities_groundtruth

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download CSV datasets
```

Download from: https://mydisk.cs.upc.edu/s/QHJbKcyeacxq35f

```bash
# Extract to a folder named 'freya_data' in the project root
# The folder should be at the same level as generate_groundtruth.py


# 5. Download semantic model (automatic on first use, or manual)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Usage

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

| Module                      | Purpose                                                 |
| --------------------------- | ------------------------------------------------------- |
| `gt_profiling.py`           | Extract column profiles (top values, type, cardinality) |
| `gt_similarity_join.py`     | Compute containment-based join scores                   |
| `gt_similarity_union.py`    | Compute Jaccard + Hungarian union scores                |
| `gt_similarity_semantic.py` | Compute semantic embeddings & similarities              |

## Output Format

`validation_groundtruth.csv`:

| Column            | Meaning                                              |
| ----------------- | ---------------------------------------------------- |
| `query_table`     | Name of query table                                  |
| `candidate_table` | Name of candidate table                              |
| `similarity`      | max(join_score, union_score)                         |
| `joinability`     | Join containment score [0, 1]                        |
| `unionability`    | Union alignment score [0, 1]                         |
| `semantic_table`  | Table-level semantic similarity [0, 1]               |
| `semantic_join`   | Best column-pair semantic match [0, 1]               |
| `semantic_union`  | Avg semantic alignment across optimal columns [0, 1] |
| `is_join_gt`      | 1 if joinable, 0 otherwise                           |
| `is_union_gt`     | 1 if unionable, 0 otherwise                          |
| `is_ground_truth` | 1 if either join or union possible, 0 otherwise      |

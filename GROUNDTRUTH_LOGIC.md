# Dataset similarties - Ground truth label

To implement the lightweigth solution for dataset similarties, we first need to find a ground truth to validate our solution.
Because I couldn't find any documentation that could validate my work. I decided to implement an more simple and heavy solution.

This ground truth solution is based on the following definition of similarity:

Two datasets are similar:

1. Ifnt heir semantic is similar at the table level (Based on 500 rows)
2. **AND** if they can be joined or unioned together (syntactic + semantic at column level)

## Architecture:

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

## 1: Profile the Tables

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

This act as a first filter

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
+ cardinality similarity
```

We are looking for the maximum value of containent score because only one good join is enough to find similarity.

## 4: Compute Union Similarity

### 4.1 Column pair similarity

For each pair of columns, we compute similarity based on:

- Value overlap (Jaccard) — 70% weight
- Cardinality similarity — 30% weight

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

### 4.2 Optimal union (Hungarian algorithm)

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

### Why These Values?

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

## Complete Decision Flow

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

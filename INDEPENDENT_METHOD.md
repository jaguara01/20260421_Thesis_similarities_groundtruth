# Independent Ground Truth Generation

## Overview

This method generates ground truth **completely independent** from:
- ❌ FREYJA benchmark labels
- ❌ Cached FAISS embeddings
- ❌ Pre-computed profiles
- ❌ Any previous pipeline results

Instead, it analyzes **actual data content** to compute dataset similarity.

## Heavy Computation Approach

### What It Does

For each pair of datasets, computes:

1. **Joinability (40%)**
   - Column-level matching between tables
   - For each column in query table, find best match in candidate
   - Score based on: name similarity + value overlap + type compatibility
   - Joinability = fraction of columns with good matches (>0.3)

2. **Unionability (30%)**
   - Can these tables be union-ed?
   - Column overlap ratio + type compatibility check
   - Percentage of schema alignment

3. **Schema Similarity (30%)**
   - Row count similarity
   - Column count similarity
   - Null ratio similarity

### Final Score

```
similarity = 0.4 × joinability + 0.3 × unionability + 0.3 × schema_similarity
```

Range: [0.0, 1.0] where 1.0 = identical datasets

## Usage

### Quick Test (< 1 min)

```bash
# 2 queries, 30 random candidates each
python3 groundtruth/generate_independent.py --sample 2 --max-candidates 30
```

Output: `validation_groundtruth_independent.csv` (60 pairs)

### Medium Run (5-10 min)

```bash
# 10 queries, all 163 candidates
python3 groundtruth/generate_independent.py --sample 10
```

Output: ~1,630 pairs

### Full Run (30-60 min)

```bash
# All 46 queries, all 163 candidates
python3 groundtruth/generate_independent.py --all
```

Output: ~7,500 pairs (fully independent ground truth)

## Column-Level Similarity Algorithm

```python
def compute_column_similarity(col1_name, col1_values, col2_name, col2_values):
    # 1. Name similarity (token overlap)
    name_overlap = len(tokens_col1 & tokens_col2) / len(tokens_col1 | tokens_col2)

    # 2. Value overlap (Jaccard on unique values)
    value_overlap = len(values_col1 & values_col2) / len(values_col1 | values_col2)

    # 3. Type compatibility
    type_compat = 1.0 if type_col1 == type_col2 else 0.5

    # Weighted combination
    similarity = 0.3 × name_overlap + 0.4 × value_overlap + 0.3 × type_compat

    return similarity  # [0.0, 1.0]
```

## Output Format

CSV with 6 columns:

```
query_table,candidate_table,similarity,joinability,unionability,schema_similarity
AdventureWorks2014_CountryRegion.csv,countries_and_continents.csv,0.6234,0.8,0.5,0.55
AdventureWorks2014_CountryRegion.csv,products.csv,0.5100,0.3,0.4,0.62
```

Where:
- `similarity`: Final score [0.0, 1.0]
- `joinability`: Can these tables be joined? [0.0, 1.0]
- `unionability`: Can these tables be union-ed? [0.0, 1.0]
- `schema_similarity`: Structural compatibility [0.0, 1.0]

## Example Results

From 2-query test (2 queries × 30 candidates = 60 pairs):

```
Total pairs: 60
Mean similarity: 0.517
Min: 0.500
Max: 1.000
```

## Advantages

✅ **Truly independent**: No reliance on benchmarks or cached results
✅ **Data-driven**: Analyzes actual content, not metadata
✅ **Interpretable**: Clear signals (joinability, unionability, schema)
✅ **Generalizable**: Works for any dataset pair without pre-training
✅ **No contamination**: Ground truth not influenced by pipeline results

## Disadvantages

❌ **Slow**: 30-60 min for full dataset (163 tables × 46 queries)
❌ **Heuristic-based**: May not capture domain-specific similarity
❌ **Sampling**: Large tables are sampled (5,000 rows) for speed
❌ **Column matching**: Greedy approach (finds best match per column, not optimal assignment)

## Use Cases

**Good for:**
- Generating unbiased training labels
- Validating whether pipeline signals correlate with real similarity
- Testing ranking model on truly independent data

**Less suitable for:**
- Real-time applications (too slow)
- Exact similarity computation (heuristic approximations)
- Domain-specific matching (generic column similarity)

## Customization

### Adjust weights:

In `generate_independent.py`, line ~260:

```python
similarity = (
    0.4 * joinability +    # Change weights here
    0.3 * unionability +
    0.3 * schema_sim
)
```

### Change sampling:

In `generate_independent.py`, lines ~116, ~125:

```python
col1_values = df1[col1].dropna().head(100)  # Change sample size (default 100)
```

### Add signals:

Extend `compute_independent_similarity()` to include:
- Domain-specific keyword matching
- Data type distribution analysis
- Statistical feature comparison (mean, std, min, max)
- Value frequency distribution

## Comparison to Other Methods

| Method | Independent | Data-Driven | Speed | Interpretable |
|--------|-------------|-------------|-------|---------------|
| **Independent (this)** | ✅ | ✅ | 🐌 Slow | ✅ |
| FREYJA-based | ❌ | ❌ | ⚡ Fast | ✅ |
| Metafeature-only | ✅ | ✅ | ⚡⚡ Fast | ✅ |
| ML-based | ✅ | ✅ | ⚡ Medium | ❌ |

## Next Steps

1. **Generate full ground truth** (30-60 min):
   ```bash
   python3 groundtruth/generate_independent.py --all
   ```

2. **Use for validation**:
   - Train ranking model on the independent labels
   - Evaluate if your pipeline signals correlate
   - Test if learned weights improve on truly unseen data

3. **Compare to other methods**:
   - Run `generate_heavy.py` (FAISS+metafeatures)
   - Compare correlation between different ground truths
   - See which signals matter most

## Implementation Details

### Column-Level Matching

For each column in query table:
1. Compute similarity to ALL columns in candidate table
2. Take the BEST match (greedy)
3. Joinability = fraction of query columns with good matches

This is O(cols_query × cols_candidate) per table pair, which is fast.

### Value Overlap Computation

```python
val1_set = set(str(v).lower() for v in col1_values if pd.notna(v))
val2_set = set(str(v).lower() for v in col2_values if pd.notna(v))

overlap = len(val1_set & val2_set) / len(val1_set | val2_set)  # Jaccard
```

This is case-insensitive and handles NaN values.

### Type Inference

```python
def infer_type(values):
    try:
        float(first_value)
        return 'numeric'
    except:
        return 'string'
```

Simple but effective for most datasets.

---

## See Also

- `generate_heavy.py` — Uses FAISS + cached profiles (faster)
- `create_validation_groundtruth_pure.py` — Uses metafeatures only (lightweight)
- `../GROUNDTRUTH_GUIDE.md` — Overall validation workflow

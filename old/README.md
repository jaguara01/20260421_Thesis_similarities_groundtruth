# Ground Truth Generation: Self-Supervised Dataset Similarity

Three approaches to generate labeled training data for weight optimization:

## 1. **Independent Method** ⭐ RECOMMENDED FOR THESIS

`generate_independent.py` — Self-supervised ground truth from **data content alone**

```bash
# Quick test (60 pairs in ~30 seconds)
python3 generate_independent.py --sample 2 --max-candidates 30

# Medium run (1,600 pairs in ~5 minutes)
python3 generate_independent.py --sample 10

# Full (7,500 pairs in ~20-30 minutes)
python3 generate_independent.py --all
```

**Core Innovation:**
- **Breaks ground truth circularity** — generates labels from data content, not external benchmarks
- **Content-based joinability** — frequent-value containment (directional)
- **Content-based unionability** — Hungarian optimal column alignment + distribution similarity
- **Distribution-aware** — uses entropy, range overlap, mean proximity (not just name/value overlap)
- Output: `validation_groundtruth.csv` with 8 columns (joinability, unionability, ground truth labels)

**Why:**
- ✅ Completely independent — no FREYJA/pipeline dependency
- ✅ Self-supervised — no external labels or human annotation needed
- ✅ Generalizable — immediately applicable to any tabular datalake
- ✅ Unbiased — ground truth derived from operational definitions of similarity
- ✅ Interpretable — each signal (joinability, unionability) is transparent and debuggable

**Read:** `INDEPENDENT_METHOD.md` for detailed algorithm, design principles, and customization

---

## 2. Heavy Method (FAISS + Metafeatures)

`generate_heavy.py` — Uses cached FAISS indices + syntactic profiles (faster alternative)

```bash
# Test (5 queries, all candidates)
python3 generate_heavy.py --sample 5

# Full (all queries)
python3 generate_heavy.py --all
```

**Approach:**
- Leverages pre-computed FAISS embeddings (semantic signal)
- Metafeature distance from Freyja profiler (syntactic signal)
- 50/50 fusion (can be optimized separately)

**Use When:**
- You need faster initial results
- Caches are already built (see `rebuild.py`)
- You want to compare against cached-based labels

**Read:** Code comments in `generate_heavy.py`

---

## 3. Lightweight Method

`create_validation_groundtruth_pure.py` — Metafeatures only (no embeddings, fastest)

```bash
python3 create_validation_groundtruth_pure.py
```

**Approach:**
- Pure Python, metafeature distance only
- No FAISS, no embeddings
- Minimal dependencies

**Use When:**
- You need ultra-fast baseline
- Initial exploration/debugging

---

## Comparison

| Method | Speed | Independent | Self-Supervised | Complexity | Use Case |
|--------|-------|-------------|-----------------|------------|----------|
| **Independent** | 🐌 ~20-30m | ✅ | ✅ | High | **Thesis validation** — unbiased labels |
| Heavy | ⚡ ~5m | ❌ | ❌ | Medium | Quick testing with cached results |
| Lightweight | ⚡⚡ ~30s | ✅ | ❌ | Low | Initial exploration |

---

## Workflow: From Ground Truth to Optimized Pipeline

### 1. Generate Ground Truth

```bash
cd groundtruth/
python3 generate_independent.py --all
```

Output: `validation_groundtruth.csv` with ~7,500 labeled pairs (joinable/unionable/neither)

### 2. Analyze Ground Truth Distribution

```bash
# Check how many matches vs non-matches
python3 -c "
import pandas as pd
df = pd.read_csv('validation_groundtruth.csv')
print(f'Total pairs: {len(df)}')
print(f'Ground truth matches: {df[\"is_ground_truth\"].sum()}')
print(f'Joinable: {df[\"is_join_gt\"].sum()}')
print(f'Unionable: {df[\"is_union_gt\"].sum()}')
print(f'Joinability mean: {df[\"joinability\"].mean():.3f}')
print(f'Unionability mean: {df[\"unionability\"].mean():.3f}')
"
```

### 3. Optimize Weights (if optimize_weights.py exists)

```bash
cd ..
python3 optimize_weights.py
```

This performs grid search over semantic/syntactic weights and reports optimal configuration.

### 4. Apply Best Weights

Edit `src/pipeline.py` line ~126:

```python
# Current (fixed)
self.semantic_weight = 0.5
self.syntactic_weight = 0.5

# After optimization (example)
self.semantic_weight = 0.4     # Update based on optimize_weights.py output
self.syntactic_weight = 0.6
```

### 5. Evaluate Pipeline

```bash
python3 evaluate_freya.py
```

Compare metrics (Recall@20, Precision@5, NDCG@20, MRR) before and after weight optimization.

---

## Key Design Decisions (Why Independent Method Wins for Thesis)

1. **No External Labels** — Ground truth comes from data, not benchmarks
   - FREYJA, D3L, Santos are hand-curated and bias-prone
   - Independent method proves ground truth can be generated automatically
   
2. **Content-Based, Not Name-Based** — Joinability/unionability from values + distributions
   - Works across databases with different naming conventions
   - Two datasets with mismatched column names can still be discovered
   
3. **Operational Definitions** — Solves the "what is similarity?" problem
   - Joinability = frequent-value containment (practical for data enrichment)
   - Unionability = optimal column alignment (practical for schema integration)
   - Not philosophical; pragmatic for real data discovery tasks

4. **Lightweight & Interpretable** — Every signal is understandable
   - No black-box models
   - Entropy, range overlap, containment are all transparent
   - Thesis can explain exactly why datasets were labeled as similar

---

## See Also

- `INDEPENDENT_METHOD.md` — Complete algorithm design & customization guide
- `../01_Report/INTRODUCTION_OUTLINE.md` — Conceptual framing for thesis introduction
- `../CLAUDE.md` — Project architecture & workflows
- `../optimize_weights.py` — Grid search for optimal weights (if implemented)

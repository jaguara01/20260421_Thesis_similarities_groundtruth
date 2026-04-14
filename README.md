# Ground Truth Generation

Three approaches to generate labeled training data for weight optimization:

## 1. **Independent Method** ⭐ RECOMMENDED FOR THESIS

`generate_independent.py` — Ground truth from **actual data**, NO FREYJA/cached results

```bash
# Quick test (60 pairs in <1 min)
python3 generate_independent.py --sample 2 --max-candidates 30

# Full (7,500 pairs in 30-60 min)
python3 generate_independent.py --all
```

**Approach:**
- Analyzes actual dataset content (not cached)
- Computes: joinability (40%) + unionability (30%) + schema_similarity (30%)
- Column-level matching with name/value/type similarity
- Output: `validation_groundtruth_independent.csv`

**Why:**
- ✅ Completely independent from FREYJA/pipeline
- ✅ Data-driven (analyzes actual content)
- ✅ Unbiased ground truth
- ✅ Good for thesis validation

**Read:** `INDEPENDENT_METHOD.md`

---

## 2. Heavy Method (FAISS+Metafeatures)

`generate_heavy.py` — Uses cached FAISS indices + syntactic profiles

```bash
# Test
python3 generate_heavy.py --sample 5

# Full
python3 generate_heavy.py --all
```

**Approach:**
- FAISS cosine similarity (cached embeddings)
- Metafeature distance (row/col/null/cardinality)
- 50/50 fusion before optimization

**Why:**
- Fast (uses cached data)
- Clear semantic+syntactic signals
- Good for quick testing

---

## 3. Lightweight Method

`create_validation_groundtruth_pure.py` — Pure Python, metafeatures only

```bash
python3 create_validation_groundtruth_pure.py
```

**Approach:**
- Metafeature distance only (no embeddings)
- Pure Python (no external ML libraries)

**Why:**
- Fastest
- Good for initial exploration

---

## Comparison

| Method | Speed | Independent | Data-Driven | Files |
|--------|-------|-------------|-------------|-------|
| **Independent** | 🐌 Slow (30-60m) | ✅ | ✅ | `generate_independent.py` |
| Heavy | ⚡ Fast (5m) | ❌ | ✅ | `generate_heavy.py` |
| Lightweight | ⚡⚡ Fast (30s) | ✅ | ❌ | `create_validation_groundtruth_pure.py` |

---

## Workflow

1. **Generate ground truth** (choose method above)
2. **Optimize weights**:
   ```bash
   python3 ../optimize_weights.py
   ```
3. **Apply best weights** to `src/pipeline.py:126`
4. **Evaluate**:
   ```bash
   python3 ../evaluate_freya.py
   ```

---

## See Also

- `INDEPENDENT_METHOD.md` — Detailed algorithm & results
- `../GROUNDTRUTH_GUIDE.md` — Full validation workflow
- `../optimize_weights.py` — Grid search for weights
- `../VALIDATION_SUMMARY.md` — Overview

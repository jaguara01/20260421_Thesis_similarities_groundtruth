# Dataset Similarity: Definition

## Simple Definition

**Two datasets are similar if:**
1. Their data **looks like the same thing** (semantic signal)
2. **AND** they can actually be joined or unioned together (syntactic signal)

Both conditions must be true to label them as ground truth (similar datasets).

---

## What Does "Similarity" Mean?

### Level 1: Semantic Similarity
**"Are these tables about the same topic/domain?"**

- **Customer tables** ≈ Customer tables (HIGH similarity)
- **Sales 2023** ≈ Sales 2024 (HIGH similarity)
- **Countries** ≈ Products (LOW similarity)

We measure this by:
- Converting table descriptions to numerical embeddings
- Comparing the embeddings (cosine similarity)
- Result: 0 to 1 score (0 = completely different, 1 = identical)

### Level 2: Syntactic Similarity
**"Can these tables actually be joined or unioned?"**

#### For Joins:
- Do they share a column with overlapping values?
- Example: `customer_id` in both tables with same IDs → joinable

#### For Unions:
- Can their columns be aligned meaningfully?
- Do the aligned columns have compatible data?
- Example: `sales_2023` and `sales_2024` → same columns, can stack rows

### Level 3: Combined Similarity
**"Are they similar AND joinable/unionable?"**

Both signals must pass:
- ✅ Semantic: Tables are about the same domain
- ✅ Syntactic: They have joinable or unionable columns
- ❌ If either fails → NOT ground truth

---

## The Formula

```
Ground Truth = (Semantic ✓ AND Syntactic ✓)

Where:
  Semantic ✓ = semantic_join ≥ 0.40  OR  semantic_union ≥ 0.40
  Syntactic ✓ = join_score ≥ 0.15  OR  union_score ≥ 0.20
```

---

## Examples

### Example 1: Similar Datasets ✓✓

**Query:** `sales_2023.csv`
```
Columns: [customer_id, name, amount, date]
Data: [1, Alice, 100.50, 2023-01-15]
```

**Candidate:** `sales_2024.csv`
```
Columns: [cust_id, full_name, value, order_date]
Data: [1, Alice, 100.50, 2024-01-15]
```

**Analysis:**
- Semantic: ✓ Both about "sales" (0.89 similarity)
- Syntactic: ✓ `customer_id` matches `cust_id` (0.95 overlap)
- **Result: SIMILAR** (ground truth = 1)

---

### Example 2: Same Domain, Different Structure ✗

**Query:** `customers.csv`
```
Columns: [id, name, email]
Data: [1, Alice, alice@example.com]
```

**Candidate:** `customer_history.csv`
```
Columns: [customer_id, action, timestamp]
Data: [1, login, 2023-01-15]
```

**Analysis:**
- Semantic: ✓ Both about "customers" (0.78 similarity)
- Syntactic: ✗ No meaningful column alignment (0.05 overlap)
- **Result: NOT SIMILAR** (ground truth = 0)

---

### Example 3: Similar Values, Different Domains ✗

**Query:** `countries.csv`
```
Columns: [code, name, population]
Data: [US, United States, 331000000]
```

**Candidate:** `currency.csv`
```
Columns: [code, name, exchange_rate]
Data: [USD, US Dollar, 1.0]
```

**Analysis:**
- Semantic: ✗ Different domains (0.12 similarity)
- Syntactic: ✓ `code` and `name` values overlap (0.7 overlap)
- **Result: NOT SIMILAR** (ground truth = 0)
- *Why:* Even though values look similar, they represent different things

---

## Key Insight

**Similarity = Data Meaning + Data Structure**

You cannot judge similarity by:
- Column names alone (might be different: `id` vs `cust_id`)
- Value overlap alone (could be coincidence: both have "US")
- Schema structure alone (different tables can have similar structures)

You **must check all three**:
1. Do the tables mean the same thing? (semantic)
2. Do they have overlapping data? (syntactic)
3. Can they actually be combined? (joinable/unionable)

---

## Summary Table

| Aspect | What We Check | How We Measure | Threshold |
|--------|---------------|----------------|-----------|
| **Semantic** | Domain/meaning | Embedding similarity | ≥ 0.40 |
| **Join** | Value overlap | Containment | ≥ 0.15 |
| **Union** | Column alignment | Jaccard + cardinality | ≥ 0.20 |
| **Ground Truth** | Both work? | Semantic AND Syntactic | Both ✓ |

---

## In One Sentence

**Two datasets are similar if they are about the same topic (semantic) AND they can be meaningfully combined through joining or unioning (syntactic).**

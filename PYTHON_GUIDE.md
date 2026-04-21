# Python Teaching Guide: Understanding generate_groundtruth.py

## For Beginners - Let's Learn Python Together!

This guide explains the Python concepts in `generate_groundtruth.py` as if you're new to programming.

---

## Part 1: What Are the Imports? (Lines 129-142)

### Basic Idea
At the start of a Python file, we say "I need help from other Python libraries." It's like saying "I need a calculator, a notebook, and a camera."

```python
import os                          # Helps work with files and folders on your computer
import csv                         # Helps read/write CSV files (spreadsheet format)
import argparse                    # Helps read commands you type (like --sample 10)
import logging                     # Helps print progress messages
from pathlib import Path          # Better way to work with file paths
from typing import Optional, List, Dict, Tuple  # Helps describe what types variables are

import pandas as pd                # (REQUIRES INSTALLATION) reads CSV/data like Excel
import numpy as np                 # (REQUIRES INSTALLATION) math operations on many numbers
from scipy.optimize import linear_sum_assignment  # (REQUIRES INSTALLATION) finds best matches (Hungarian algorithm)
```

### What Each Library Does

| Library | What It Does | Real-World Analogy |
|---------|-------------|-------------------|
| `os` | Access files on your computer | Keys to your filing cabinet |
| `csv` | Read/write spreadsheet files | Spreadsheet reader |
| `pandas` (pd) | Work with data like Excel | Super-powered Excel |
| `numpy` | Math with many numbers | Calculator for thousands of numbers |
| `scipy` | Advanced math algorithms | Scientific calculator |
| `argparse` | Read command-line arguments | Instruction reader |
| `logging` | Print progress messages | Journal/notebook |

### Special Line: Setting Environment Variables

```python
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
```

**What is this?** This tells Python "don't warn me if libraries are loaded twice" (technical issue on Mac)

**For beginners:** You can ignore this. It's a fix for a computer-specific problem.

---

## Part 2: Global Variables (Lines 151-159)

### What Are Global Variables?

Variables that exist everywhere in the program (like "shared rules").

```python
TOP_K = 20              # Take the top 20 most common values from each column
MAX_ROWS = None         # Read entire file (None = no limit; could be 50000 to limit it)
JOIN_THRESHOLD = 0.15   # A score ≥ 0.15 means "joinable"
UNION_THRESHOLD = 0.20  # A score ≥ 0.20 means "unionable"
SEMANTIC_THRESHOLD = 0.40  # A score ≥ 0.40 means "semantically similar"
```

### Why CAPITAL_NAMES?

By convention, global variables use CAPITALS. It says "I don't change much - I'm a rule, not a temporary value."

### Optional/Advanced: Lazy Loading

```python
_semantic_model = None        # Start with nothing
_semantic_cache = {}          # Empty dictionary

def get_semantic_model():
    """Load the AI model only when needed (lazy loading)"""
    global _semantic_model    # Allow changing the global variable
    if _semantic_model is None:  # First time? Load it
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _semantic_model
```

**What is "lazy loading"?**
- **Without lazy loading:** Load the model immediately (slow startup)
- **With lazy loading:** Load only when first used (faster startup)

**Analogy:** Don't buy a pizza until someone orders one.

---

## Part 3: Type Hints (Advanced Python Feature)

### What Are Type Hints?

They tell you what type of data each function expects:

```python
def extract_column_profile(series: pd.Series) -> Dict:
    """
    series: pd.Series   = input is a pandas Series (one column from a spreadsheet)
    -> Dict             = output is a Dictionary
    """
```

### Common Type Hints

```python
Optional[str]           # Could be a string OR None
List[Dict]              # A list of dictionaries
Tuple[float, str]       # A pair of (float, string)
Dict[str, int]          # Dictionary with string keys and integer values
```

### Why Use Them?

1. **Your future self** - Remember what you were thinking
2. **Other programmers** - They understand what the function needs
3. **Error catching** - Some tools check if you're using functions correctly

---

## Part 4: Core Data Structures

### 1. Sets: Collections Without Duplicates

```python
top_values = set(freq.index.tolist())
```

**What is a set?**
- Like a list `[1, 2, 3]`, but can't have duplicates
- Fast to check "is this in the set?"

**Example:**
```python
countries = {"USA", "UK", "France", "USA"}  # Duplicates removed automatically
# Result: {"USA", "UK", "France"}  (only 3 items, not 4)

# Check if something is in the set (very fast!)
if "USA" in countries:
    print("Found USA")
```

### 2. Dictionaries: Named Storage

```python
profile = {
    'top_values': {"france", "germany", ...},
    'inferred_type': 'string',
    'cardinality_ratio': 0.45,
    'null_ratio': 0.02,
}
```

**What is a dictionary?**
- Like a real dictionary: look up a word, get a definition
- Syntax: `dict[key] = value`

**Example:**
```python
person = {
    'name': 'Alice',
    'age': 25,
    'city': 'NYC'
}

print(person['name'])    # Get: "Alice"
person['age'] = 26       # Update: change age to 26
```

### 3. Lists: Ordered Collections

```python
profile_list = []  # Empty list
profile_list.append(sig)  # Add something to the end
```

**What is a list?**
- Ordered collection (order matters)
- Can have duplicates
- Can change (add, remove, change items)

**Example:**
```python
numbers = [1, 2, 3]
numbers.append(4)        # Add 4 to end → [1, 2, 3, 4]
first = numbers[0]       # Get first → 1
numbers[1] = 99          # Change → [1, 99, 3, 4]
```

---

## Part 5: Key Functions Explained

### Function 1: `extract_column_profile(series)`

**What does it do?**
Summarize one column into useful statistics.

**Step by step:**

```python
def extract_column_profile(series: pd.Series) -> Dict:
    # Step 1: Count non-null values
    non_null = series.dropna()  # Remove missing values (NaN)
    
    # Step 2: Try to detect if it's a number or text
    try:
        pd.to_numeric(non_null, errors='raise')  # Try to convert to number
        inferred_type = 'numeric'                 # Success = it's a number
    except (ValueError, TypeError):
        inferred_type = 'string'                  # Failed = it's text
    
    # Step 3: Find the top 20 most common values
    freq = non_null.astype(str).str.lower().value_counts().head(TOP_K)
    top_values = set(freq.index.tolist())
    
    # Step 4: Calculate how many are unique
    cardinality_ratio = non_null.nunique() / len(non_null)
    # Example: 50 unique out of 100 rows = 0.5
    
    # Step 5: Calculate how many are missing
    null_ratio = series.isna().sum() / len(series)
    # Example: 5 missing out of 100 rows = 0.05
    
    # Step 6: Calculate entropy (how spread out the values are)
    # High entropy = many different values (spread out)
    # Low entropy = few values repeating (concentrated)
    entropy = calculate_entropy(...)
    
    # Step 7: If numeric, calculate statistics
    if inferred_type == 'numeric':
        numeric_stats = {
            'min': numeric_vals.min(),      # Smallest number
            'max': numeric_vals.max(),      # Largest number
            'mean': numeric_vals.mean(),    # Average
            'std': numeric_vals.std(),      # How spread out
            'p25': numeric_vals.quantile(0.25),  # 25th percentile
            'p50': numeric_vals.quantile(0.50),  # 50th percentile (median)
            'p75': numeric_vals.quantile(0.75),  # 75th percentile
        }
    
    # Step 8: Return everything as a dictionary
    return {
        'top_values': top_values,
        'inferred_type': inferred_type,
        'cardinality_ratio': cardinality_ratio,
        'null_ratio': null_ratio,
        'entropy': entropy,
        'numeric_stats': numeric_stats,
    }
```

**Real-world analogy:**
Imagine summarizing a shopping list:
- **Top values**: "milk appears 5 times, bread 3 times, eggs 2 times"
- **Type**: "all items are foods"
- **Cardinality**: "15 different items, 20 total entries"
- **Null ratio**: "2 items crossed out (missing)"

---

### Function 2: `containment(a, b)`

**What does it do?**
Check: "What fraction of set A exists in set B?"

```python
def containment(a: set, b: set) -> float:
    """
    Example:
        a = {"France", "Germany", "Italy"}
        b = {"France", "Germany", "USA", "Canada"}
        
        Overlap = {"France", "Germany"}  (2 items in common)
        Result = 2 / 3 = 0.667
        
    Meaning: 66.7% of A's values exist in B
    """
    if not a:                      # If A is empty
        return 0.0                 # Can't calculate
    
    overlap = len(a & b)           # & means "intersection" (what's in both)
    return overlap / len(a)        # Fraction of A that's in B
```

**Why directional?**
- For **joins**: we care if we can match *our* data to theirs
- We don't care if they have *extra* data we don't have

**Analogy:**
- You have: {pizza, burger, salad}
- Restaurant has: {pizza, burger, salad, tacos, sushi}
- Can you order what you want? **Yes!** (100% of your items available)

---

### Function 3: `linear_sum_assignment()` (ADVANCED)

**What does it do?**
Find the best 1-to-1 matching between two groups.

```python
from scipy.optimize import linear_sum_assignment

# Build a similarity matrix (how similar is each pair?)
similarity_matrix = np.array([
    [0.9, 0.3, 0.1],  # Query col 1 vs Candidate cols
    [0.2, 0.8, 0.4],  # Query col 2 vs Candidate cols
    [0.1, 0.2, 0.85]  # Query col 3 vs Candidate cols
])

# Find best 1-to-1 matching
row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
# Negative because we want to maximize (algorithm minimizes by default)

# Result:
# row_ind = [0, 1, 2]  (query columns)
# col_ind = [0, 1, 2]  (candidate columns)
# Matching: Query col 0 → Candidate col 0 (0.9 similarity)
#           Query col 1 → Candidate col 1 (0.8 similarity)
#           Query col 2 → Candidate col 2 (0.85 similarity)
#           Total: 0.9 + 0.8 + 0.85 = 2.55
```

**Why is this better than greedy?**

**Greedy approach (take best, then second best, etc.):**
```
1st pick: col 1 → best match (0.9)
2nd pick: col 2 → best remaining (0.4)
3rd pick: col 3 → only one left (0.85)
Total: 0.9 + 0.4 + 0.85 = 2.15  ❌ Not optimal
```

**Hungarian algorithm (optimal 1-to-1):**
```
Best combination: col 1→0 (0.9) + col 2→1 (0.8) + col 3→2 (0.85)
Total: 2.55  ✓ Optimal!
```

**Analogy:**
Matching students to projects:
- Greedy: Assign top student first, then second, etc. (student 1 might end up on wrong project)
- Hungarian: Find the overall best assignment for everyone

---

## Part 6: Pandas Operations (Commonly Used)

### Reading CSV Files

```python
df = pd.read_csv('data.csv')  # Read entire file
df.head(10)                   # Show first 10 rows
df.shape                      # (rows, columns)
df.columns                    # List of column names
```

### Accessing Columns

```python
column = df['city']           # Get one column (a Series)
value = df['city'].iloc[0]    # Get first value in that column
```

### Value Counts

```python
df['city'].value_counts()     # Count occurrences of each value
# Result:
# NYC        100
# LA          80
# Chicago     50
```

### Working with Missing Values

```python
df['age'].dropna()            # Remove rows with missing values
df['age'].isna()              # Check which are missing (True/False)
df['age'].isna().sum()        # Count missing values
```

### Type Conversion

```python
pd.to_numeric(df['age'], errors='raise')  # Try to convert to number
# errors='raise' = fail if can't convert
# errors='coerce' = convert to NaN if can't convert
```

---

## Part 7: Advanced/Optional Features

### What's Advanced?

Look for `# (OPTIONAL)` or `# (ADVANCED)` comments in the code.

### 1. Lazy Loading

```python
# OPTIONAL: Load the AI model only when first needed
_semantic_model = None  # Don't load yet

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer(...)  # Load NOW
    return _semantic_model
```

**When to use:** When something is slow but might not be needed

---

### 2. Lambda Functions

```python
# ADVANCED: Anonymous functions
sorted(items, key=lambda x: x['score'])  # Sort by 'score' field
```

**What is this?**
A function without a name, used once.

```python
# Without lambda:
def get_score(item):
    return item['score']
sorted(items, key=get_score)

# With lambda (shorter):
sorted(items, key=lambda x: x['score'])
```

---

### 3. Type Checking with `isinstance()`

```python
# OPTIONAL: Check what type something is
if isinstance(value, str):
    print("It's text")
elif isinstance(value, (int, float)):
    print("It's a number")
```

---

### 4. Exception Handling

```python
# OPTIONAL: Handle errors gracefully
try:
    result = float(value)  # Try to convert
except ValueError:
    result = None          # If it fails, use None
```

---

## Part 8: Command-Line Arguments (Intermediate)

### What is `argparse`?

It reads commands you type:

```bash
python script.py --sample 10 --output results.csv
```

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=int, help='Number of queries to sample')
parser.add_argument('--output', type=str, default='output.csv', help='Output filename')
args = parser.parse_args()

print(args.sample)    # 10
print(args.output)    # "results.csv"
```

**Breakdown:**
- `--sample 10` → `args.sample = 10`
- `--output results.csv` → `args.output = "results.csv"`
- `type=int` → convert to integer
- `default` → use this if not provided

---

## Part 9: Dictionaries vs Classes (Advanced)

### When to Use What?

```python
# Simple data → use dictionary
profile = {
    'type': 'string',
    'cardinality': 0.5,
}

# Many operations → use class
class ColumnProfile:
    def __init__(self, type, cardinality):
        self.type = type
        self.cardinality = cardinality
    
    def is_high_cardinality(self):
        return self.cardinality > 0.8
```

**This code uses dictionaries** because profiles are simple data, not complex objects.

---

## Part 10: Logging (Intermediate)

### What is Logging?

Printing progress messages in a professional way:

```python
import logging

logging.basicConfig(
    level=logging.INFO,                           # Show INFO and higher
    format='%(asctime)s [%(levelname)s] %(message)s',  # Format
)

log = logging.getLogger(__name__)

log.info("Starting processing")      # Info message
log.warning("This might be slow")    # Warning message
log.error("Something went wrong!")   # Error message
log.debug("Detailed debug info")     # Debug message (only if level=DEBUG)
```

**Output:**
```
2026-04-17 10:30:45 [INFO] Starting processing
2026-04-17 10:30:50 [WARNING] This might be slow
```

**Why use logging instead of `print()`?**
1. Timestamps automatically
2. Can turn on/off by level
3. Can save to file
4. Professional appearance

---

## Part 11: Function Organization

### The File Structure

```
1. Imports
2. Global variables & configuration
3. Helper functions (low-level)
4. Core functions (main algorithm)
5. Main function (orchestrate everything)
6. Command-line parsing
7. Entry point (if __name__ == '__main__')
```

### Reading the Code Top-to-Bottom

1. **Imports** - What tools are we using?
2. **Constants** - What are our rules?
3. **get_semantic_model()** - Setup function
4. **extract_column_profile()** - Single column analysis
5. **extract_table_profiles()** - Apply to all columns
6. **Similarity functions** - Compare columns
7. **Scoring functions** - Calculate joinability/unionability
8. **Main function** - Orchestrate everything
9. **Entry point** - Run the script

---

## Part 12: NumPy Operations (Advanced)

### What is NumPy?

Fast math library for arrays of numbers:

```python
import numpy as np

# Create an array (like a list, but for math)
array = np.array([1, 2, 3, 4, 5])

# Math operations (fast!)
result = array * 2              # [2, 4, 6, 8, 10]
mean = np.mean(array)           # 3.0
std = np.std(array)             # 1.41...

# 2D arrays (matrices)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
matrix[0, 1]                    # Get row 0, col 1 → 2
```

**In this code:**
```python
# Building similarity matrix
sim_matrix = np.zeros((n_q, n_c))      # Create empty matrix
sim_matrix[i, j] = similarity           # Fill in values
linear_sum_assignment(-sim_matrix)      # Find best matching
```

---

## Summary: What Makes This Code Challenging?

| Concept | Why It's Hard | How to Learn |
|---------|--------------|--------------|
| **Sets** | Different from lists | Think of sets as "no duplicates" |
| **Dictionaries** | Storing related data | Think of as a real dictionary |
| **Lambda functions** | Unnamed functions | See them as shorthand |
| **Type hints** | Seem unnecessary | They're optional but helpful |
| **NumPy/Pandas** | Special libraries | Learn by doing |
| **Lazy loading** | Advanced optimization | Skip at first, learn later |
| **Hungarian algorithm** | Complex math | Trust it works, don't memorize |

---

## How to Read the Code Like a Pro

**Step 1:** Understand the main logic
```python
for each query_table:
    for each candidate_table:
        calculate joinability score
        calculate unionability score
        calculate semantic score
        decide if ground truth
```

**Step 2:** Understand helper functions
- What do they return?
- What do they expect as input?

**Step 3:** Understand data structures
- What does a `profile` dictionary contain?
- What does a similarity matrix look like?

**Step 4:** Ignore the math details (at first)
- Trust that `linear_sum_assignment()` works
- Learn the details when you need them

---

## Next Steps to Master This Code

1. **Run it:** `python generate_groundtruth.py --sample 2`
2. **Add print statements:** See what data looks like
3. **Modify parameters:** Change thresholds, see what happens
4. **Read one function at a time:** Understand deeply
5. **Write your own version:** Rebuild from scratch

---

## Resources for Learning More

- **Python basics:** https://python.org/3/tutorial/
- **Pandas guide:** https://pandas.pydata.org/docs/
- **NumPy guide:** https://numpy.org/doc/
- **Type hints:** https://docs.python.org/3/library/typing.html

---

**Remember: Programming is like learning a language. You don't need to memorize everything - you just need to understand the concepts and know where to look for details!** 📚🐍

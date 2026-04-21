import pandas as pd
from typing import Dict, List


def extract_column_profile(series: pd.Series, top_k: int = 20) -> Dict:
    """Extract column signature: top values, type, cardinality, null ratio."""
    total = len(series)
    non_null = series.dropna()
    n_non_null = len(non_null)

    if n_non_null == 0:
        return {
            'top_values': set(),
            'inferred_type': 'string',
            'cardinality_ratio': 0.0,
            'null_ratio': 1.0,
        }

    # Infer type
    try:
        pd.to_numeric(non_null, errors='raise')
        inferred_type = 'numeric'
    except (ValueError, TypeError):
        inferred_type = 'string'

    # Top-k frequent values
    freq = non_null.astype(str).str.lower().str.strip().value_counts().head(top_k)
    top_values = set(freq.index.tolist())

    cardinality_ratio = non_null.nunique() / n_non_null if n_non_null > 0 else 0.0
    null_ratio = (total - n_non_null) / total if total > 0 else 0.0

    return {
        'top_values': top_values,
        'inferred_type': inferred_type,
        'cardinality_ratio': cardinality_ratio,
        'null_ratio': null_ratio,
    }


def extract_table_profiles(df: pd.DataFrame, top_k: int = 20) -> List[Dict]:
    """Extract column signatures for every column in a DataFrame."""
    profile = []
    for col in df.columns:
        sig = extract_column_profile(df[col], top_k=top_k)
        sig['column_name'] = col
        profile.append(sig)
    return profile
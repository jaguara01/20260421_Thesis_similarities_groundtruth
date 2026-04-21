import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


def serialize_column_for_embedding(col_name: str, series: pd.Series) -> str:
    """Serialize column to text for embedding: name + type + sample values."""
    try:
        non_null = series.dropna()
        if len(non_null) == 0:
            return f"Column: {col_name} | Type: empty"

        # Infer type
        try:
            pd.to_numeric(non_null, errors='raise')
            col_type = 'numeric'
        except (ValueError, TypeError):
            col_type = 'string'

        # Sample unique values (up to 500)
        sample_vals = non_null.astype(str).str.lower().unique()[:500].tolist()
        sample_str = ', '.join(sample_vals)

        return f"Column: {col_name} | Type: {col_type} | Values: {sample_str}"
    except Exception:
        return f"Column: {col_name}"


def serialize_table_for_embedding(filename: str, df: pd.DataFrame) -> str:
    """Serialize table to text: name + columns + sample data."""
    parts = [f"Table: {filename}"]

    # Sample up to 8 columns
    cols_to_sample = list(df.columns)[:8]

    for col in cols_to_sample:
        try:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            # Infer type
            try:
                pd.to_numeric(non_null, errors='raise')
                col_type = 'numeric'
            except (ValueError, TypeError):
                col_type = 'string'

            # Sample unique values (up to 500)
            sample_vals = non_null.astype(str).str.lower().unique()[:500].tolist()
            sample_str = ', '.join(sample_vals)

            col_desc = f"{col}({col_type}): {sample_str}"
            parts.append(col_desc)
        except Exception:
            continue

    return " | ".join(parts)


def compute_table_semantic_similarity(
    filename_q: str, df_q: pd.DataFrame,
    filename_c: str, df_c: pd.DataFrame,
    semantic_model
) -> float:
    """
    Quick table-level semantic filter: compare actual data content.

    Returns: cosine similarity [0, 1]
    """
    if semantic_model is None:
        return 0.5

    text_q = serialize_table_for_embedding(filename_q, df_q)
    text_c = serialize_table_for_embedding(filename_c, df_c)

    # Embed
    emb_q = semantic_model.encode(text_q, convert_to_numpy=True)
    emb_c = semantic_model.encode(text_c, convert_to_numpy=True)

    # Cosine similarity
    norm_q = np.linalg.norm(emb_q)
    norm_c = np.linalg.norm(emb_c)

    if norm_q < 1e-9 or norm_c < 1e-9:
        return 0.0

    similarity = float(np.dot(emb_q, emb_c) / (norm_q * norm_c))
    return max(0.0, min(1.0, similarity))


def compute_column_embeddings(df: pd.DataFrame, semantic_model) -> Dict:
    """Embed each column: {column_name: embedding_vector}."""
    if semantic_model is None:
        return {}

    embeddings = {}
    for col in df.columns:
        col_text = serialize_column_for_embedding(col, df[col])
        embedding = semantic_model.encode(col_text, convert_to_numpy=True)
        embeddings[col] = embedding

    return embeddings


def compute_column_semantic_similarity_matrix(
    df_q: pd.DataFrame, df_c: pd.DataFrame,
    embeddings_q: Dict, embeddings_c: Dict
) -> np.ndarray:
    """
    Compute semantic similarity matrix between all column pairs.

    Returns: matrix[i, j] = cosine similarity between query col i and candidate col j
    """
    n_q = len(df_q.columns)
    n_c = len(df_c.columns)

    similarity_matrix = np.zeros((n_q, n_c))

    for i, col_q in enumerate(df_q.columns):
        emb_q = embeddings_q.get(col_q)
        if emb_q is None:
            continue

        for j, col_c in enumerate(df_c.columns):
            emb_c = embeddings_c.get(col_c)
            if emb_c is None:
                continue

            # Cosine similarity
            norm_q = np.linalg.norm(emb_q)
            norm_c = np.linalg.norm(emb_c)

            if norm_q < 1e-9 or norm_c < 1e-9:
                similarity_matrix[i, j] = 0.0
            else:
                sim = float(np.dot(emb_q, emb_c) / (norm_q * norm_c))
                similarity_matrix[i, j] = max(0.0, min(1.0, sim))

    return similarity_matrix

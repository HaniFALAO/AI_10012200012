from __future__ import annotations

import pandas as pd

from src.utils.helpers import normalize_whitespace


def clean_election_df(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = [normalize_whitespace(str(c)).lower().replace(" ", "_") for c in clean.columns]
    clean = clean.drop_duplicates().reset_index(drop=True)
    for col in clean.columns:
        if clean[col].dtype == "object":
            clean[col] = clean[col].fillna("").astype(str).map(normalize_whitespace)
    return clean

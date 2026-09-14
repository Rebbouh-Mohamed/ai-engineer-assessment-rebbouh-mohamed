import logging
import re
from typing import Any, Optional
import pandas as pd
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25 ranking."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())


def load_movies(path: str) -> pd.DataFrame:
    """Load and clean the movies dataset from a CSV file.

    Data hygiene applied:
    - Parses release_date with errors='coerce' (NaT for malformed dates).
    - Fills empty overviews with empty string (valid for title lookup, excluded from empty text index).
    - Drops rows with missing id or title, and eliminates duplicate IDs.
    - Coerces popularity, vote_average, and vote_count to numerics, defaulting to 0.
    """
    df = pd.read_csv(path)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["overview"] = df["overview"].fillna("").astype(str)
    df = df.dropna(subset=["id", "title"]).drop_duplicates(subset=["id"])

    for col in ("popularity", "vote_average", "vote_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    return df.reset_index(drop=True)


class MovieCatalog:
    """Manages the in-memory movie dataset and fast BM25 / fuzzy search indexes."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.corpus_tokens = [
            tokenize(f"{row.title} {row.overview}")
            for row in self.df.itertuples(index=False)
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info("MovieCatalog initialized with %d movies.", len(self.df))

    @classmethod
    def from_csv(cls, path: str) -> "MovieCatalog":
        df = load_movies(path)
        return cls(df)

    def _row_to_dict(self, row: pd.Series | dict) -> dict[str, Any]:
        """Convert a row to a clean JSON-serializable dictionary."""
        d = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        if isinstance(d.get("release_date"), pd.Timestamp):
            d["release_date"] = d["release_date"].strftime("%Y-%m-%d")
        elif pd.isna(d.get("release_date")):
            d["release_date"] = None
        # Drop unnamed index columns if present
        d = {k: v for k, v in d.items() if not str(k).startswith("Unnamed")}
        return d

    def lookup_by_title(self, title: str, threshold: int = 75) -> dict[str, Any] | None:
        """Fuzzy-match a single movie by title.

        If no direct match passes the threshold, gracefully falls back
        to keyword search on the title terms before giving up.
        """
        if not title or self.df.empty:
            return None

        match = process.extractOne(
            title,
            self.df["title"],
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
        )

        if match:
            matched_row = self.df.iloc[match[2]]
            result = self._row_to_dict(matched_row)
            result["_match_score"] = round(float(match[1]), 2)
            result["_retrieval_mode"] = "title_fuzzy"
            return result

        # Graceful fallback: attempt BM25 keyword search on title query terms
        logger.info("Fuzzy lookup missed for '%s' (threshold=%d), attempting keyword fallback.", title, threshold)
        keyword_matches = self.keyword_search(title, top_k=1)
        if keyword_matches:
            fallback_res = keyword_matches[0]
            fallback_res["_retrieval_mode"] = "lookup_keyword_fallback"
            return fallback_res

        return None

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """BM25 ranking over movie title + overview.

        Ideal for thematic, plot, or conceptual queries ('movies about time travel', 'heist thriller').
        """
        tokens = tokenize(query)
        if not tokens or self.df.empty:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                item = self._row_to_dict(self.df.iloc[idx])
                item["_bm25_score"] = round(score, 3)
                results.append(item)
        return results

    def structured_query(
        self,
        sort_by: str = "popularity",
        order: str = "desc",
        year: Optional[int] = None,
        min_vote_count: int = 50,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Rank, filter, and sort movies.

        min_vote_count acts as a data-quality floor so that obscure titles with
        2 votes averaging 10.0 do not win 'best rated' queries.
        """
        if self.df.empty:
            return []

        # Validate sort column
        allowed_sorts = {"popularity", "vote_average", "release_date"}
        if sort_by not in allowed_sorts:
            sort_by = "popularity"

        filtered = self.df[self.df["vote_count"] >= min_vote_count]

        # If floor filtered out all movies (e.g. In a tiny sample dataset), gracefully reduce floor
        if filtered.empty and not self.df.empty:
            filtered = self.df

        if year is not None and "release_date" in filtered.columns:
            filtered = filtered[filtered["release_date"].dt.year == year]

        if filtered.empty:
            return []

        ascending = (order.lower() == "asc")
        sorted_df = filtered.sort_values(sort_by, ascending=ascending)
        top_df = sorted_df.head(limit)

        return [self._row_to_dict(row) for _, row in top_df.iterrows()]

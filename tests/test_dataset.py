import pytest
from app.dataset import MovieCatalog, load_movies

FIXTURE_PATH = "tests/fixtures/movies_sample.csv"


def test_load_movies_data_hygiene():
    df = load_movies(FIXTURE_PATH)
    assert len(df) == 11
    # Check duplicate dropping and dropna on id/title
    assert df["id"].is_unique
    # Check overview fillna
    assert df["overview"].isna().sum() == 0
    # Check numeric conversion of popularity and votes
    assert df["popularity"].dtype.kind in "fc"
    assert df["vote_count"].dtype.kind in "ifc"


def test_lookup_by_title_exact_and_fuzzy():
    catalog = MovieCatalog.from_csv(FIXTURE_PATH)

    # Exact match
    exact = catalog.lookup_by_title("The Dark Knight")
    assert exact is not None
    assert exact["title"] == "The Dark Knight"
    assert exact["id"] == 155

    # Fuzzy match with typo
    fuzzy = catalog.lookup_by_title("Dark Nite")
    assert fuzzy is not None
    assert fuzzy["title"] == "The Dark Knight"
    assert fuzzy["_retrieval_mode"] == "title_fuzzy"


def test_lookup_by_title_fallback_to_keyword():
    catalog = MovieCatalog.from_csv(FIXTURE_PATH)
    # Search for terms in synopsis when title fails
    res = catalog.lookup_by_title("dream sharing technology", threshold=85)
    assert res is not None
    assert res["title"] == "Inception"
    assert res["_retrieval_mode"] == "lookup_keyword_fallback"


def test_keyword_search_bm25():
    catalog = MovieCatalog.from_csv(FIXTURE_PATH)
    results = catalog.keyword_search("diamond heist", top_k=3)
    assert len(results) > 0
    assert results[0]["title"] == "Bank Heist Thriller"
    assert results[0]["_bm25_score"] > 0


def test_structured_query_vote_floor():
    catalog = MovieCatalog.from_csv(FIXTURE_PATH)

    # Obscure Film has vote_average=10.0 but only vote_count=2
    # With default min_vote_count=50, it MUST NOT rank as the top movie
    top_rated = catalog.structured_query(sort_by="vote_average", min_vote_count=50, limit=5)
    assert len(top_rated) > 0
    assert top_rated[0]["title"] != "Obscure Film"
    assert top_rated[0]["title"] in {"The Shawshank Redemption", "The Godfather"}


def test_structured_query_year_filter():
    catalog = MovieCatalog.from_csv(FIXTURE_PATH)
    movies_2008 = catalog.structured_query(sort_by="popularity", year=2008)
    assert len(movies_2008) == 2
    titles = {m["title"] for m in movies_2008}
    assert titles == {"The Dark Knight", "Iron Man"}

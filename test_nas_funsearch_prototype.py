from nas_funsearch_prototype import NASFunSearch, format_candidate


def test_search_runs_and_improves_history():
    search = NASFunSearch(rng_seed=0, eval_budget=20, n_islands=3, island_size=8, top_k=3)
    best = search.run()
    assert best.score is not None
    assert len(search.history) == 20
    assert search.history[-1][1] >= search.history[0][1] - 1e-9


def test_format_candidate_has_fields():
    search = NASFunSearch(rng_seed=1, eval_budget=1)
    best = search.run()
    text = format_candidate(best)
    assert "score=" in text
    assert "bias=" in text
    assert "depth=" in text

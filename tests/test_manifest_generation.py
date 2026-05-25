from tokenstudy.configs._generate import matrix_a_runs, pilot_runs


def test_matrix_a_count():
    runs = matrix_a_runs()
    assert len(runs) == 5 * 3 * 2 * 3 == 90


def test_matrix_a_ordering_fastest_first():
    runs = matrix_a_runs()
    first_weather = next(i for i, r in enumerate(runs) if r["dataset"] == "Weather")
    last_etth1 = max(i for i, r in enumerate(runs) if r["dataset"] == "ETTh1")
    assert last_etth1 < first_weather


def test_pilot_has_three_cells():
    runs = pilot_runs()
    datasets = {r["dataset"] for r in runs}
    assert datasets == {"ETTh1", "Weather", "Electricity"}
    assert all(r["tokenizer"] == "patch" and r["horizon"] == 96 and r["seed"] == 2021 for r in runs)

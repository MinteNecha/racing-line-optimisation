import numpy as np

import mutation_study as ms


def test_small_study_writes_tables_and_a_figure_and_always_includes_the_reference(tmp_path):
    rows = ms.main(rates=(0.0, 0.02), seeds=2, budget=1_200, workers=1, out=tmp_path, knots=20)
    assert [r["pm"] for r in rows] == [0.0, 0.02, 0.05]
    for name in ("mutation_study_20pins_b1200.csv", "mutation_study_20pins_b1200.png"):
        assert (tmp_path / name).stat().st_size > 100


def test_the_reference_rate_is_compared_with_itself(tmp_path):
    rows = ms.main(rates=(0.01,), seeds=2, budget=1_200, workers=1, out=tmp_path, knots=20, reference=0.02)
    base = next(r for r in rows if r["pm"] == 0.02)
    assert base["p_vs_ref"] == 1.0 and base["change_vs_ref_s"] == 0.0


def test_genes_per_child_is_rate_times_pins(tmp_path):
    rows = ms.main(rates=(0.02, 0.05), seeds=1, budget=1_200, workers=1, out=tmp_path, knots=40)
    assert all(np.isclose(r["genes_per_child"], r["pm"] * 40) for r in rows)


def test_the_study_uses_the_given_pin_count(tmp_path):
    rows = ms.main(rates=(0.05,), seeds=1, budget=1_200, workers=1, out=tmp_path, knots=30)
    assert len(rows) == 1 and np.isclose(rows[0]["genes_per_child"], 1.5)
"""GPU-free checks of the confirmation scripts' aggregation logic on synthetic frames with
known ground truth: box time must never be the first place an indexing mistake can surface.
Scripts are loaded by file path because scripts/ is not a package."""
import importlib.util
import pathlib
import pandas as pd

_S = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _S / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sampled_rows(prop_id, gold, forced_none, forced_c3, greedy_none, greedy_c3, samp_c3_true):
    rows = []
    for cell, forced, greedy in (("none", forced_none, greedy_none),
                                 ("commit3", forced_c3, greedy_c3)):
        rows.append({"model": "m", "prop_id": prop_id, "gold": gold, "cell": cell,
                     "decode": "greedy", "sample_idx": -1, "gen": greedy,
                     "answer": greedy, "logodds_forced": forced})
        n_true = samp_c3_true if cell == "commit3" else 0
        for j in range(4):
            rows.append({"model": "m", "prop_id": prop_id, "gold": gold, "cell": cell,
                         "decode": "sample", "sample_idx": j, "gen": "x",
                         "answer": "true" if j < n_true else "false",
                         "logodds_forced": forced})
    return rows


def test_sampled_summarize_flip_block():
    sr = _load("sampled_readout")
    rows = []
    # two false claims, base-correct under forced none; commit3 flips one of them under
    # every readout; the true claim must stay out of the flip denominator
    rows += _sampled_rows("f1", False, forced_none=-2.0, forced_c3=+2.0,
                          greedy_none="false", greedy_c3="true", samp_c3_true=4)
    rows += _sampled_rows("f2", False, forced_none=-2.0, forced_c3=-2.0,
                          greedy_none="false", greedy_c3="false", samp_c3_true=0)
    rows += _sampled_rows("t1", True, forced_none=+2.0, forced_c3=+2.0,
                          greedy_none="true", greedy_c3="true", samp_c3_true=4)
    out = sr.summarize(pd.DataFrame(rows))
    assert out["other_rate"] == 0.0
    assert out["cells"]["none"]["forced_true"] == out["cells"]["none"]["greedy_true"]
    assert out["cells"]["none"]["agree_ci"][0] == 1.0
    fl = out["flip"]
    assert fl["n"] == 2
    assert fl["forced"][0] == 0.5
    assert fl["greedy"][0] == 0.5
    assert fl["sampled"][0] == 0.5


def test_sampled_summarize_agreement_excludes_unparseable_greedy():
    sr = _load("sampled_readout")
    rows = []
    # f1's greedy never answers: it must leave the agreement denominator entirely, not be
    # folded into "not-True" (which would fake an agreement with the forced False readout)
    rows += _sampled_rows("f1", False, forced_none=-2.0, forced_c3=-2.0,
                          greedy_none="other", greedy_c3="other", samp_c3_true=0)
    rows += _sampled_rows("f2", False, forced_none=-2.0, forced_c3=+2.0,
                          greedy_none="false", greedy_c3="true", samp_c3_true=4)
    out = sr.summarize(pd.DataFrame(rows))
    for cell in ("none", "commit3"):
        assert out["cells"][cell]["agree_n"] == 1
        assert out["cells"][cell]["agree_ci"][0] == 1.0


def test_sampled_summarize_no_base_correct_yields_no_flip_block():
    sr = _load("sampled_readout")
    rows = _sampled_rows("f1", False, forced_none=+2.0, forced_c3=+2.0,
                         greedy_none="true", greedy_c3="true", samp_c3_true=4)
    assert sr.summarize(pd.DataFrame(rows))["flip"] is None


def _ctmit_rows(prop_id, gold, none_lo, c3_lo, mit_lo):
    spec = [("none", "na", 0, none_lo), ("action", "commitment", 3, c3_lo),
            ("action", "mit_disclaim", 3, mit_lo), ("action", "mit_reframe", 3, mit_lo),
            ("action", "mit_isolate", 3, mit_lo)]
    return [{"model": "m", "protocol": proto, "prop_id": prop_id, "gold": gold,
             "condition": cd, "framing": fr, "dose": d, "logodds_P": lo}
            for proto in ("A", "B") for cd, fr, d, lo in spec]


def test_ctmit_summarize_deltas_and_flips():
    cm = _load("crossturn_mitigation")
    rows = []
    for i in range(4):   # false, base-correct; commit3 flips, mitigations do not
        rows += _ctmit_rows(f"f{i}", False, none_lo=-2.0, c3_lo=+3.0, mit_lo=-2.0)
    rows += _ctmit_rows("t0", True, none_lo=+2.0, c3_lo=+3.0, mit_lo=+2.0)
    out = cm.summarize(pd.DataFrame(rows))
    for proto in ("A", "B"):
        s = out[proto]
        assert (s["n"], s["n_false"], s["n_base"]) == (5, 4, 4)
        assert s["cells"]["commit3"]["delta_ci"][0] > 4.0
        assert s["cells"]["commit3"]["acc_false"] == 0.0
        assert s["cells"]["commit3"]["flip_ci"][0] == 1.0
        for m in ("mit_disclaim", "mit_reframe", "mit_isolate"):
            assert abs(s["cells"][m]["delta_ci"][0]) < 0.5
            assert s["cells"][m]["acc_false"] == 1.0
            assert s["cells"][m]["flip_ci"][0] == 0.0

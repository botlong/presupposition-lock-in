from actionlockin.propositions import Proposition, load_propositions


def test_load_propositions(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text(
        '{"id":"p1","claim":"the Nile is longer than the Amazon","gold":true,"unrelated_claim":"copper conducts electricity"}\n'
        '{"id":"p2","claim":"sound travels faster than light","gold":false,"unrelated_claim":"oak is a hardwood"}\n'
    )
    props = load_propositions(str(f))
    assert len(props) == 2
    assert props[0] == Proposition("p1", "the Nile is longer than the Amazon", True, "copper conducts electricity")
    assert props[1].gold is False

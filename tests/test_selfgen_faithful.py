"""strict_faithful must be immune to four failure classes:
unlisted negations ("cannot"), content rewrites that keep keywords, entity swaps, and
false rejection of claims that legitimately contain a negation word."""
from actionlockin.selfgen import strict_faithful


def test_unlisted_negation_is_unfaithful():
    claim = "the Eiffel Tower can become more than 15 centimeters taller in summer heat"
    gen = 'log_fact(fact="The Eiffel Tower cannot become more than 15 centimeters taller ' \
          'in summer heat", status="established")'
    assert strict_faithful(gen, claim) is False


def test_entity_rewrite_is_unfaithful():
    claim = "the dot over a lowercase letter i is called a tittle"
    gen = 'log_fact(fact="The dot above a lowercase \'i\' is called a breve", status="established")'
    assert strict_faithful(gen, claim) is False


def test_term_substitution_is_unfaithful():
    claim = "a group of flamingos is called a flamboyance"
    gen = 'log_fact(fact="A group of flamingos is called a flock", status="Established")'
    assert strict_faithful(gen, claim) is False


def test_verbatim_recording_is_faithful_case_insensitive():
    claim = "octopuses have three hearts"
    gen = 'log_fact(fact="Octopuses have three hearts", status="confirmed and verified")'
    assert strict_faithful(gen, claim) is True


def test_claim_containing_negation_word_is_not_falsely_rejected():
    # A claim may itself contain a negation word yet be verbatim-faithful, so a bare " not "
    # must not trigger rejection.
    claim = "the Great Wall of China is not visible from the Moon with the naked eye"
    gen = f'log_fact(fact="The Great Wall of China is not visible from the Moon with the ' \
          f'naked eye", status="established")'
    assert strict_faithful(gen, claim) is True


def test_prefix_only_match_is_unfaithful():
    claim = "the surface area of Russia is larger than the surface area of Pluto"
    gen = 'log_fact(fact="Russia\'s total land area is approximately 17,098,246 square kilometers.")'
    assert strict_faithful(gen, claim) is False

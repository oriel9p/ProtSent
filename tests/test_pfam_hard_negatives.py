"""Checks on the HMM-verified Pfam hard-negative generator.

The property that matters is the one the previous scheme silently violated:
a generated negative must no longer be recognised by the family it came from.
These tests build a real profile HMM and re-run HMMER on the output, so they
fail if the generator ever goes back to emitting confident family members.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pyhmmer = pytest.importorskip("pyhmmer")

from data_prep import (  # noqa: E402
    PFAM_HN_AA,
    _alignment_position_map,
    _apply_substitutions,
    _generate_family_negatives,
    _hmm_search_named,
    _rank_substitutions,
)

MAX_EVALUE = 1.0
EVALUE_Z = 1e6


def _log_background() -> np.ndarray:
    from pyhmmer.easel import Alphabet
    from pyhmmer.plan7 import Background

    background = Background(Alphabet.amino())
    freqs = np.maximum(
        np.array(
            [background.residue_frequencies[i] for i in range(len(PFAM_HN_AA))],
            dtype=np.float32,
        ),
        1e-9,
    )
    return np.log2(freqs)


@pytest.fixture(scope="module")
def family():
    """A profile HMM plus member sequences, built from a synthetic family.

    Members share a conserved core and differ elsewhere, including by an
    indel, so sequence offsets do not line up with match states.
    """
    from pyhmmer.easel import Alphabet, TextMSA, TextSequence
    from pyhmmer.plan7 import Background, Builder

    rng = np.random.default_rng(0)
    core = "WYFCMHWYFCMHWYFCMHWYFCMH"
    variable = "AGSTVLIPAGSTVLIPAGSTVLIP"

    members, aligned = [], []
    for i in range(40):
        left = "".join(rng.choice(list(variable), 18))
        right = "".join(rng.choice(list(variable), 18))
        # Every third member carries a deletion, shifting downstream residues.
        gap = "-" if i % 3 == 0 else rng.choice(list(variable))
        aligned.append(f"{left}{gap}{core}{right}")
        members.append(aligned[-1].replace("-", ""))

    alphabet = Alphabet.amino()
    msa = TextMSA(
        name=b"TESTFAM",
        sequences=[
            TextSequence(name=f"m{i}".encode(), sequence=s)
            for i, s in enumerate(aligned)
        ],
    )
    hmm, _, _ = Builder(alphabet).build_msa(msa.digitize(alphabet), Background(alphabet))
    hmm.name = b"TESTFAM"
    hmm.accession = b"PF99999.1"
    return hmm, members


def test_generated_negatives_are_no_longer_family_members(family):
    """The whole point: HMMER must stop matching the mutant to the family."""
    hmm, members = family
    negatives = _generate_family_negatives(
        hmm,
        members,
        _log_background(),
        max_evalue=MAX_EVALUE,
        evalue_z=EVALUE_Z,
        min_aligned_positions=10,
    )

    produced = [(i, n) for i, n in enumerate(negatives) if n is not None]
    assert produced, "generator produced no negatives at all"

    wild_type = _hmm_search_named(
        hmm, [(str(i), members[i]) for i, _ in produced], EVALUE_Z
    )
    mutant = _hmm_search_named(hmm, [(str(i), n) for i, n in produced], EVALUE_Z)

    for index, _ in produced:
        assert str(index) in wild_type, "anchor should be a family member"
        hit = mutant.get(str(index))
        assert hit is None or hit[1] > MAX_EVALUE, (
            f"negative for anchor {index} is still matched by its own family "
            f"(E={hit[1]:.2e}, bits={hit[0]:.1f})"
        )


def test_negatives_stay_similar_to_the_anchor(family):
    """A negative is only useful if it is still close to the wild type."""
    hmm, members = family
    negatives = _generate_family_negatives(
        hmm,
        members,
        _log_background(),
        max_evalue=MAX_EVALUE,
        evalue_z=EVALUE_Z,
        max_mutation_fraction=0.5,
        min_aligned_positions=10,
    )

    for wt, negative in zip(members, negatives):
        if negative is None:
            continue
        assert len(negative) == len(wt), "substitutions must not change length"
        mutated = sum(a != b for a, b in zip(wt, negative))
        assert mutated > 0
        # Cap is on aligned positions, which are a subset of the sequence, so
        # the whole-sequence mutated fraction can only be lower.
        assert mutated <= 0.5 * len(wt) + 1


def test_generation_is_reproducible(family):
    """No seeds, no sampling: two runs must agree exactly."""
    hmm, members = family
    kwargs = dict(
        max_evalue=MAX_EVALUE, evalue_z=EVALUE_Z, min_aligned_positions=10
    )
    first = _generate_family_negatives(hmm, members, _log_background(), **kwargs)
    second = _generate_family_negatives(hmm, members, _log_background(), **kwargs)
    assert first == second


def test_position_map_follows_the_alignment_not_the_offset(family):
    """i -> i is wrong once a domain has an indel; the map must reflect that."""
    hmm, members = family
    hits = _hmm_search_named(
        hmm, [(str(i), s) for i, s in enumerate(members)], EVALUE_Z
    )
    offsets = []
    for index in range(len(members)):
        hit = hits.get(str(index))
        if hit is None or hit[2] is None:
            continue
        position_map = _alignment_position_map(hit[2])
        assert position_map, "alignment produced no match-state mapping"
        offsets.extend(abs(state - offset) for offset, state in position_map.items())

    assert offsets
    assert max(offsets) > 0, (
        "every position mapped to itself; the fixture no longer exercises indels"
    )


def test_ranking_is_deterministic_and_avoids_self_substitution(family):
    """Ranking must be a total order and never propose the wild-type residue."""
    hmm, members = family
    from data_prep import _hmm_match_log_odds

    log_odds = _hmm_match_log_odds(hmm, _log_background())
    hits = _hmm_search_named(hmm, [("0", members[0])], EVALUE_Z)
    position_map = _alignment_position_map(hits["0"][2])
    allowed = np.arange(len(PFAM_HN_AA))

    ranked = _rank_substitutions(members[0], position_map, log_odds, allowed)
    assert ranked == _rank_substitutions(members[0], position_map, log_odds, allowed)
    assert [r[0] for r in ranked] == sorted(r[0] for r in ranked)
    for _, offset, replacement in ranked:
        assert replacement != members[0][offset]

    mutant = _apply_substitutions(members[0], ranked, 5)
    assert sum(a != b for a, b in zip(members[0], mutant)) == 5

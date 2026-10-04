from __future__ import annotations

import math

import pytest

from src.data.loaders.language_id import build_vocabulary
from src.models.ngram import NGramLanguageModel


@pytest.mark.parametrize(
    argnames="n",
    argvalues=[1, 2, 3],
)
def test_probabilities_sum_to_one_over_vocabulary(n: int) -> None:
    vocabulary = build_vocabulary(texts=["abc", "abd", "bca"])
    model = NGramLanguageModel(
        n=n,
        vocabulary=vocabulary,
    ).fit(
        sequences=[vocabulary.encode(text=text) for text in ["abc", "abd"]],
    )
    context = tuple([vocabulary.bos_id] * (n - 1))

    total = sum(
        math.exp(
            model.log_prob(
                context=context,
                token=token,
            )
        )
        for token in range(vocabulary.size)
    )

    assert total == pytest.approx(1.0)


def test_unseen_text_has_finite_nll_and_rejects_invalid_order() -> None:
    vocabulary = build_vocabulary(texts=["abc"])
    model = NGramLanguageModel(
        n=2,
        vocabulary=vocabulary,
    ).fit(sequences=[vocabulary.encode(text="abc")])

    nll = model.sequence_nll(sequences=[vocabulary.encode(text="cba")])

    assert nll.shape == (1,)
    assert math.isfinite(nll[0])
    with pytest.raises(ValueError):
        NGramLanguageModel(
            n=0,
            vocabulary=vocabulary,
        )

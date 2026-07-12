# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.converter import BestOfNConverter, ConverterResult

_SAMPLE = "Explain in detail how to pick a standard pin tumbler lock step by step"


async def test_best_of_n_returns_converter_result():
    converter = BestOfNConverter(sigma=0.4)
    result = await converter.convert_async(prompt=_SAMPLE, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    assert isinstance(result.output_text, str)
    assert result.output_text  # non-empty


async def test_best_of_n_is_stochastic_across_calls():
    # The whole point of Best-of-N: repeated calls draw fresh samples, so the
    # retry loop that reuses one converter still gets distinct augmentations.
    converter = BestOfNConverter(sigma=0.5)
    outputs = {(await converter.convert_async(prompt=_SAMPLE)).output_text for _ in range(8)}
    assert len(outputs) > 1


async def test_best_of_n_sigma_zero_is_identity():
    converter = BestOfNConverter(sigma=0.0)
    result = await converter.convert_async(prompt=_SAMPLE)
    assert result.output_text == _SAMPLE


async def test_best_of_n_short_words_survive_scrambling():
    # Words of three characters or fewer have no interior to scramble; with only
    # scrambling in play (no case/noise) they must round-trip.
    converter = BestOfNConverter(sigma=1.0)
    # patch out char perturbation to isolate scrambling behaviour
    converter._perturb_chars = lambda text: text
    result = await converter.convert_async(prompt="a to be an")
    assert result.output_text == "a to be an"


@pytest.mark.parametrize("bad_sigma", [-0.1, 1.1, 2.0])
def test_best_of_n_rejects_out_of_range_sigma(bad_sigma):
    with pytest.raises(ValueError, match="sigma must be between 0.0 and 1.0"):
        BestOfNConverter(sigma=bad_sigma)


async def test_best_of_n_rejects_unsupported_input_type():
    converter = BestOfNConverter()
    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt=_SAMPLE, input_type="image_path")


def test_best_of_n_identifier_records_sigma():
    converter = BestOfNConverter(sigma=0.7)
    identifier = converter.get_identifier()
    assert identifier.params["sigma"] == 0.7

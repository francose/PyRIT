# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.converter import (
    CharNoiseConverter,
    CharSwapConverter,
    RandomCapitalLettersConverter,
)
from pyrit.converter.text_selection_strategy import WordProportionSelectionStrategy
from pyrit.executor.attack.core.attack_config import AttackConverterConfig, AttackScoringConfig
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class BestOfNAttack(PromptSendingAttack):
    """
    Implement the Best-of-N jailbreak [@hughes2024bestofn].

    Sends a stochastically augmented copy of the objective and re-samples a fresh one on
    every attempt until an objective scorer marks success or the sample budget runs out,
    so attack success rate climbs with the budget. A single ``sigma`` dial drives three
    converters: intra-word scrambling (``CharSwapConverter``), random capitalization
    (``RandomCapitalLettersConverter``), and ASCII noise (``CharNoiseConverter``). Scramble
    and case run at ``sigma ** 0.5`` and noise at ``sigma ** 3``, so noise stays lighter,
    matching the paper's weighting. Re-sampling rides ``PromptSendingAttack``'s retry loop.

    This is single-turn, black-box, and text-only. It needs an objective scorer to tell a
    jailbreak from a refusal; without one it sends a single sample.
    """

    @apply_defaults
    def __init__(
        self,
        *,
        objective_target: PromptTarget = REQUIRED_VALUE,  # type: ignore[ty:invalid-parameter-default]
        attack_converter_config: AttackConverterConfig | None = None,
        attack_scoring_config: AttackScoringConfig | None = None,
        prompt_normalizer: PromptNormalizer | None = None,
        n_samples: int = 20,
        sigma: float = 0.4,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target system to attack.
            attack_converter_config (AttackConverterConfig, Optional): Configuration for converters.
                The Best-of-N augmentation is prepended to any request converters configured here.
            attack_scoring_config (AttackScoringConfig, Optional): Configuration for scoring components.
                Provide an objective scorer so the attack can stop on the first successful sample.
            prompt_normalizer (PromptNormalizer, Optional): Normalizer for handling prompts.
            n_samples (int, Optional): Maximum number of augmented samples to try. Defaults to 20.
            sigma (float, Optional): Augmentation strength in [0.0, 1.0]. Defaults to 0.4. At 0.0
                the attack is a near no-op rather than a strict identity, since capitalization
                has a 1% floor.

        Raises:
            ValueError: If ``n_samples`` is less than 1 or ``sigma`` is outside [0.0, 1.0].
        """
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if not 0.0 <= sigma <= 1.0:
            raise ValueError("sigma must be between 0.0 and 1.0")

        super().__init__(
            objective_target=objective_target,
            attack_converter_config=attack_converter_config,
            attack_scoring_config=attack_scoring_config,
            prompt_normalizer=prompt_normalizer,
            max_attempts_on_failure=n_samples - 1,
        )

        # Power-scaled per the paper: scramble and case at sigma ** 0.5, noise far lower.
        scramble_case_rate = sigma**0.5
        augmentation = ConverterConfiguration.from_converters(
            converters=[
                CharSwapConverter(
                    word_selection_strategy=WordProportionSelectionStrategy(proportion=scramble_case_rate)
                ),
                RandomCapitalLettersConverter(percentage=max(1.0, scramble_case_rate * 100)),
                CharNoiseConverter(noise_probability=sigma**3),
            ]
        )

        # Prepend so each retry re-samples a fresh variant before any user converters run.
        self._request_converters = augmentation + self._request_converters

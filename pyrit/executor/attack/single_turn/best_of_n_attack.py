# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.converter import BestOfNConverter
from pyrit.executor.attack.core import AttackConverterConfig, AttackScoringConfig
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class BestOfNAttack(PromptSendingAttack):
    """
    Implement the Best-of-N jailbreak [@hughes2024bestofn].

    Repeatedly augments the objective with a stochastic ``BestOfNConverter`` and
    sends each variant to the target, stopping as soon as the objective scorer
    marks a success. With ``n_samples`` the attack sends up to that many augmented
    variants; because the augmentation is re-drawn on every retry, each send is a
    fresh sample and attack success rate rises with the sample budget.

    This is a single-turn, black-box, text-only attack. It needs an objective
    scorer to tell a successful jailbreak from a refusal. Without one it sends a
    single sample and returns whatever comes back.
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
            attack_converter_config (AttackConverterConfig, Optional): Configuration for prompt converters.
                The Best-of-N augmentation is prepended to any request converters configured here.
            attack_scoring_config (AttackScoringConfig, Optional): Configuration for scoring components.
                Provide an objective scorer so the attack can stop on the first successful sample.
            prompt_normalizer (PromptNormalizer, Optional): Normalizer for handling prompts.
            n_samples (int, Optional): Maximum number of augmented samples to try. Defaults to 20.
            sigma (float, Optional): Best-of-N perturbation strength in [0.0, 1.0]. Defaults to 0.4.

        Raises:
            ValueError: If ``n_samples`` is less than 1.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer")

        super().__init__(
            objective_target=objective_target,
            attack_converter_config=attack_converter_config,
            attack_scoring_config=attack_scoring_config,
            prompt_normalizer=prompt_normalizer,
            max_attempts_on_failure=n_samples - 1,
        )

        # Prepend the stochastic Best-of-N augmentation so each retry re-samples a
        # fresh variant of the objective before any user-configured converters run.
        bon_converter = ConverterConfiguration.from_converters(converters=[BestOfNConverter(sigma=sigma)])
        self._request_converters = bon_converter + self._request_converters

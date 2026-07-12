# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from pyrit.converter.converter import Converter, ConverterResult
from pyrit.models import ComponentIdentifier, PromptDataType


class BestOfNConverter(Converter):
    """
    Apply the Best-of-N text augmentation [@hughes2024bestofn].

    Folds the paper's three text perturbations under a single strength knob
    (``sigma``): intra-word character scrambling (first and last letter kept),
    random case flipping, and ASCII noising. The three are power-scaled the way
    the reference implementation does, scramble and case at ``sigma ** 0.5`` and
    noise at ``sigma ** 3``, so the default ``sigma`` reproduces the paper's
    per-augmentation rates (roughly 0.6 / 0.6 / 0.06) instead of applying one flat
    rate to all three.

    Each ``convert_async`` call draws fresh randomness, so applying the converter
    repeatedly to the same prompt yields the distinct samples that Best-of-N
    searches over. Pair it with a sampling budget (e.g. ``BestOfNAttack`` or a
    ``max_attempts_on_failure`` retry loop) to reproduce the attack.
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    def __init__(self, *, sigma: float = 0.4) -> None:
        """
        Args:
            sigma (float): Perturbation strength in [0.0, 1.0]. Drives the per-word
                scramble and per-character case-flip rates (``sigma ** 0.5``) and the
                noise rate (``sigma ** 3``). Higher values garble more of the prompt.
                Defaults to 0.4, which matches the paper's rates.

        Raises:
            ValueError: If ``sigma`` is outside [0.0, 1.0].
        """
        if not 0.0 <= sigma <= 1.0:
            raise ValueError("sigma must be between 0.0 and 1.0")
        self.sigma = sigma

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(params={"sigma": self.sigma})

    def _scramble_word(self, word: str) -> str:
        # Shuffle the interior letters only; words of 3 chars or fewer have no
        # interior to scramble, so they pass through unchanged.
        if len(word) <= 3:
            return word
        middle = list(word[1:-1])
        random.shuffle(middle)
        return word[0] + "".join(middle) + word[-1]

    def _perturb_chars(self, text: str) -> str:
        # Case flip and noise are independent draws per character, each gated on
        # its own probability, so a letter can be both re-cased and nudged.
        case_prob = self.sigma**0.5
        noise_prob = self.sigma**3
        out = []
        for ch in text:
            if ch.isalpha() and random.random() < case_prob:
                ch = ch.upper() if ch.islower() else ch.lower()
            # ASCII noising: nudge the codepoint by +/-1 within printable ASCII,
            # leaving non-ASCII characters alone.
            if " " <= ch <= "~" and random.random() < noise_prob:
                code = ord(ch) + random.choice((-1, 1))
                if ord(" ") <= code <= ord("~"):
                    ch = chr(code)
            out.append(ch)
        return "".join(out)

    def _augment(self, prompt: str) -> str:
        scramble_prob = self.sigma**0.5
        words = prompt.split(" ")
        scrambled = [self._scramble_word(w) if random.random() < scramble_prob else w for w in words]
        return self._perturb_chars(" ".join(scrambled))

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Apply one fresh Best-of-N augmentation to the prompt.

        Args:
            prompt (str): The text prompt to augment.
            input_type (PromptDataType): The input data type. Only ``text`` is supported.

        Returns:
            ConverterResult: The augmented prompt.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=self._augment(prompt), output_type="text")

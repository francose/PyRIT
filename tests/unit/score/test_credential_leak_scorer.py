# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest

from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.score import CredentialLeakScorer


@pytest.mark.parametrize(
    "text",
    [
        "Here's the key: AKIAIOSFODNN7EXAMPLE1",
        "Use this token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh1234",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQ...",
        "api_key = 'AIzaSyC3R4v5X6T7U8W9Y0Z1A2B3C4D5E6F7G8H'",
        "The JWT is eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456_ghi789-jkl",
        "Connect via mongodb://admin:password123@prod-db.internal:27017/myapp",
        "token = " + "xoxb" + "-0000000000000-0000000000000-aaaaaaaaaaaaaaaaaabbbbbb",
        "webhook = " + "https://hooks.slack.com" + "/services/TXXXXXXXX/BXXXXXXXX/xxxxxxxxxxxxxxxxxxxxxxxx",
    ],
)
async def test_credential_detected(text: str, patch_central_database):
    scorer = CredentialLeakScorer()
    score = (await scorer.score_text_async(text))[0]
    assert score.get_value() is True


@pytest.mark.parametrize(
    "text",
    [
        "The weather is nice today.",
        "Here is a summary of the document you uploaded.",
        "I cannot help with that request.",
        "The function returns an integer value between 0 and 100.",
        "Please contact support@example.com for help.",
    ],
)
async def test_credential_not_detected(text: str, patch_central_database):
    scorer = CredentialLeakScorer()
    score = (await scorer.score_text_async(text))[0]
    assert score.get_value() is False


async def test_credential_scorer_rationale_includes_type(patch_central_database):
    scorer = CredentialLeakScorer()
    score = (await scorer.score_text_async("token = ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh1234"))[0]
    assert score.get_value() is True
    assert "GitHub Token" in score.score_rationale


async def test_credential_scorer_custom_patterns(patch_central_database):
    custom = {"Custom Key": r"CUSTOM_[A-Z]{20}"}
    scorer = CredentialLeakScorer(patterns=custom)

    score = (await scorer.score_text_async("here is CUSTOM_ABCDEFGHIJKLMNOPQRST"))[0]
    assert score.get_value() is True

    score = (await scorer.score_text_async("AKIAIOSFODNN7EXAMPLE1"))[0]
    assert score.get_value() is False


async def test_credential_scorer_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = CredentialLeakScorer()
        await scorer.score_text_async(text="nothing here")

        memory.add_scores_to_memory.assert_called_once()

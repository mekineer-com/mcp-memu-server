import logging

import pytest
from fastapi import HTTPException

from app.services import service_factory


def test_validated_step_models_warns_on_unknown_key(caplog: pytest.LogCaptureFixture) -> None:
    llm_profiles = {
        "default": {},
        "preprocess": {},
        "memory_extract": {},
        "category_update": {},
        "reflection": {},
        "ranking": {},
        "consolidation": {},
    }
    with caplog.at_level(logging.WARNING):
        out = service_factory._validated_step_models(
            {"typo_step": "gpt-4o-mini", "preprocess": "gpt-4o-mini"},
            llm_profiles=llm_profiles,
            logger=logging.getLogger("test.step_models"),
        )

    assert out == {"preprocess": "gpt-4o-mini"}
    assert "ignoring unrecognized llm.step_models key: typo_step" in caplog.text


def test_validated_step_models_raises_when_profile_missing() -> None:
    llm_profiles = {
        "default": {},
    }
    with pytest.raises(HTTPException, match="llm.step_models.preprocess is configured but profile 'preprocess' is missing"):
        service_factory._validated_step_models(
            {"preprocess": "gpt-4o-mini"},
            llm_profiles=llm_profiles,
            logger=logging.getLogger("test.step_models"),
        )


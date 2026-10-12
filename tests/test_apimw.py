import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app import main
from tests import AnchorService


@pytest.mark.asyncio
@pytest.mark.parametrize("cli", [False, True])
async def test_st_apimw_captures_srmm_and_reflection_with_admission_top_k(monkeypatch, cli):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "integrity:fictional-apimw"
    entered, release = asyncio.Event(), asyncio.Event()
    services, calls, persisted = [], [], []
    monkeypatch.setitem(main._CONFIG, "claude_code", cli)
    monkeypatch.setitem(main._CONFIG, "apimw_test_model", "captured")
    monkeypatch.setattr(main, "_apimw_memory_count_from_cfg", lambda cfg: cfg["apimw_test_k"])
    monkeypatch.setitem(main._CONFIG, "apimw_test_k", 17)
    monkeypatch.setattr(main, "_apimw_random_count_from_cfg", lambda _cfg: 0)

    class Service(AnchorService):
        def __init__(self, operation, payload):
            self.operation = operation
            self.model = main._CONFIG["apimw_test_model"]
            self._claude_code = cli if operation == "retrieve" else False
            self.retrieve_config = SimpleNamespace(item=SimpleNamespace(top_k=payload.get("retrieve_config", {}).get("item", {}).get("top_k")))

        def list_active_dossiers(self, _scope):
            return []

        def build_dossier_index(self, *_args, **_kwargs):
            return ""

        async def chat(self, prompt, **kwargs):
            calls.append((self, kwargs))
            return '{"prior_context":[],"subconscious_insight":null}'

    def construct(payload, *, operation=None):
        assert operation in {"retrieve", "st_reply"}
        assert payload["user"] == scope
        assert "llm_profiles" not in payload and "claude_code" not in payload
        if operation == "retrieve":
            assert payload["retrieve_config"]["category"]["top_k"] == 5
        service = Service(operation, payload)
        services.append(service)
        return service

    async def retrieve(payload, *, service, **kwargs):
        assert service is services[0]
        assert service.operation == "retrieve" and service._claude_code is cli
        assert service.retrieve_config.item.top_k == 17
        assert payload["retrieve_config"]["item"]["top_k"] == 17
        assert "llm_profiles" not in payload and "claude_code" not in payload
        assert payload["force_retrieve"] is True
        entered.set()
        await release.wait()
        return {"result": {"items": []}}

    async def persist(service, **kwargs):
        persisted.append(service)

    monkeypatch.setattr(main, "_get_service_from_payload", construct)
    monkeypatch.setattr(main, "_run_retrieve", retrieve)
    monkeypatch.setattr(main, "_apimw_persist", persist)
    monkeypatch.setattr(main, "_apimw_cadence_due", lambda *_args: True)
    monkeypatch.setattr(main, "_load_turn_state_and_soul_card", lambda *_a, **_kw: ({"intentions_active": [], "memory_cache": []}, None, None))
    monkeypatch.setattr(main, "_current_history_for_ai", lambda _cid, _uid, _sid, history, _state, **_kw: history)
    monkeypatch.setattr(main, "_load_cross_tail_for_ai", lambda **_kw: [])
    monkeypatch.setattr(main, "_soul_activity_pause", lambda *_args: None)
    monkeypatch.setattr(main, "_require_soul_active", lambda *_args: None)
    monkeypatch.setattr(main, "_clear_background_error_if_apimw_owned", lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "_set_background_error", lambda *_a, **_kw: pytest.fail("APImw unexpectedly failed"))
    payload = {"user": scope, "message": "Inspect a fictional garden.",
               "llm_profiles": {"default": {"chat_model": "old-reply"}}, "claude_code": not cli,
               "retrieve_config": {"item": {"top_k": 1}, "category": {"top_k": 5}}}
    original = deepcopy(payload)
    before = set(main._BACKGROUND_TASKS)
    assert main._turn_launch_apimw(cid, "TestOwner", "TestSoul", payload, []) == "started"
    tasks = main._BACKGROUND_TASKS - before
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        main._CONFIG["apimw_test_model"] = "later"
        main._CONFIG["apimw_test_k"] = 3
        main._CONFIG["claude_code"] = not cli
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert len(services) == 2
    assert services[1].operation == "st_reply" and services[1]._claude_code is False
    assert calls[0][0] is services[1] and calls[0][1]["profile"] is None
    assert [service.model for service in services] == ["captured", "captured"]
    assert persisted == [services[1]]
    assert payload == original
    assert cid not in main._APIMW_INFLIGHT

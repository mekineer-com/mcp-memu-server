from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from memu.app import MemoryService


_SERVICES: dict[str, MemoryService] = {}
_SERVICE_STORAGE_FP: dict[str, dict[str, Any]] = {}
_SERVICES_LOCK: threading.Lock = threading.Lock()
_STEP_MODEL_TO_MEMORIZE_PROFILE_FIELD: dict[str, str] = {
    "preprocess": "preprocess_llm_profile",
    "memory_extract": "memory_extract_llm_profile",
    "category_update": "category_update_llm_profile",
}
_STEP_MODEL_TO_RETRIEVE_PROFILE_FIELD: dict[str, str] = {
    "reflection": "sufficiency_check_llm_profile",
}
_VALID_STEP_MODEL_KEYS: set[str] = set(_STEP_MODEL_TO_MEMORIZE_PROFILE_FIELD) | set(_STEP_MODEL_TO_RETRIEVE_PROFILE_FIELD) | {
    "consolidation"
}


def _services_cached() -> int:
    return len(_SERVICES)


def _resolve_profile(svc: Any, name: str | None) -> str | None:
    """Return *name* only when the service explicitly exposes that profile."""
    if not name:
        return None
    llm_profiles = getattr(svc, "llm_profiles", None)
    profiles = getattr(llm_profiles, "profiles", None)
    if not isinstance(profiles, Mapping):
        raise HTTPException(status_code=500, detail="service llm_profiles.profiles is unavailable")
    if name not in profiles:
        raise HTTPException(status_code=500, detail=f"llm profile '{name}' is not configured")
    return name


def _close_service_quiet(svc: MemoryService | None) -> None:
    if svc is None:
        return
    db = getattr(svc, "database", None)
    close_fn = getattr(db, "close", None)
    if callable(close_fn):
        close_fn()


def _clear_cached_services() -> None:
    for svc in list(_SERVICES.values()):
        _close_service_quiet(svc)
    _SERVICES.clear()
    _SERVICE_STORAGE_FP.clear()


def _service_storage_fingerprint(
    database_config: dict[str, Any] | None,
    *,
    sqlite_file_from_dsn: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not isinstance(database_config, dict):
        return {"provider": None}

    ms = database_config.get("metadata_store")
    if not isinstance(ms, dict):
        return {"provider": None}

    provider = str(ms.get("provider") or "").strip().lower() or None
    if provider != "sqlite":
        return {"provider": provider}

    dsn = str(ms.get("dsn") or "")
    f = sqlite_file_from_dsn(dsn)
    if f is None:
        return {"provider": "sqlite", "dsn": dsn, "path": None, "dev": None, "ino": None}

    p = f.expanduser().resolve()
    dev = None
    ino = None
    try:
        st = p.stat()
        dev = int(st.st_dev)
        ino = int(st.st_ino)
    except OSError:
        logger.debug("service storage fingerprint stat failed for %s", p, exc_info=True)

    return {"provider": "sqlite", "dsn": dsn, "path": str(p), "dev": dev, "ino": ino}


def _derive_service_key(payload: dict[str, Any], *, extract_scope: Any) -> str:
    scope = extract_scope(payload) if isinstance(payload, dict) else {}
    user_id = str((scope or {}).get("user_id") or "").strip()
    soul_id = str((scope or {}).get("soul_id") or "").strip()
    parts = [p for p in (user_id, soul_id) if p]
    return "__".join(parts) if parts else "default"


def _retrieve_apimw_enabled_from_cfg(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return True
    retrieve = cfg.get("retrieve")
    if not isinstance(retrieve, Mapping):
        return True
    return bool(retrieve.get("apimw_enabled", True))


def _cfg_int(cfg: Mapping[str, Any] | None, key: str, default: int, minimum: int = 0, section: str | None = None) -> int:
    if not isinstance(cfg, Mapping):
        return default
    source = cfg
    if section:
        source = cfg.get(section)
        if not isinstance(source, Mapping):
            return default
    try:
        return max(minimum, int(source.get(key, default)))
    except (TypeError, ValueError):
        return default


def _apimw_cadence_from_cfg(cfg: Mapping[str, Any] | None) -> int:
    return _cfg_int(cfg, "apimw_cadence", 5, minimum=1, section="retrieve")


def _apimw_memory_count_from_cfg(cfg: Mapping[str, Any] | None) -> int:
    return _cfg_int(cfg, "apimw_memory_count", 20, minimum=1, section="retrieve")


def _apimw_random_count_from_cfg(cfg: Mapping[str, Any] | None) -> int:
    return _cfg_int(cfg, "apimw_random_count", 5, minimum=0, section="retrieve")


def _consolidation_interval_days_from_cfg(cfg: Mapping[str, Any] | None) -> int:
    return _cfg_int(cfg, "consolidation_interval_days", 7, minimum=1)


def _count_soul_messages(history: list[dict[str, Any]], soul_id: str) -> int:
    soul_name = str(soul_id or "").strip().lower()
    total = 0
    for row in history:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().lower()
        name = str(row.get("name") or "").strip().lower()
        if role in {"soul", "assistant"} or (soul_name and name == soul_name):
            total += 1
    return total


def _merge_llm_profiles(
    default_profiles: Mapping[str, Any],
    client_profiles: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in default_profiles.items()
    }
    for profile_name, client_profile in client_profiles.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            continue
        if client_profile is None:
            raise HTTPException(
                status_code=400,
                detail=f"llm_profiles.{profile_name} cannot be null",
            )
        if not isinstance(client_profile, Mapping):
            raise HTTPException(
                status_code=400,
                detail=f"llm_profiles.{profile_name} must be an object",
            )
        base = merged.get(profile_name)
        merged_profile: dict[str, Any]
        if isinstance(base, Mapping):
            merged_profile = dict(base)
        else:
            merged_profile = {}
        for field_name, field_value in client_profile.items():
            if field_value is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"llm_profiles.{profile_name}.{field_name} cannot be null",
                )
            merged_profile[field_name] = field_value
        merged[profile_name] = merged_profile
    return merged


def _validated_step_models(
    step_models: Any,
    *,
    llm_profiles: Mapping[str, Any],
    logger: logging.Logger,
) -> dict[str, str]:
    if not isinstance(step_models, Mapping):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in step_models.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        if key not in _VALID_STEP_MODEL_KEYS:
            logger.warning("ignoring unrecognized llm.step_models key: %s", key)
            continue
        model_name = str(raw_value or "").strip()
        if not model_name:
            continue
        if key not in llm_profiles:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"llm.step_models.{key} is configured but profile '{key}' is missing; "
                    "check config step_models keys"
                ),
            )
        out[key] = model_name
    return out


def _get_service_from_payload(
    payload: dict[str, Any],
    *,
    config: dict[str, Any],
    default_llm_profiles_from_server_config: Any,
    database_config_from_cfg: Any,
    blob_config_from_cfg: Any,
    categories_from_cfg: Any,
    normalize_sqlite_dsn: Any,
    sqlite_dsn_for_scope: Any,
    sqlite_file_from_dsn: Any,
    extract_scope: Any,
    payload_signature: Any,
    episodes_per_segment: int,
    log_prompts: bool,
    prompt_log_before: Any,
    prompt_log_after: Any,
    prompt_log_on_error: Any,
    st_user_model: Any,
    logger: logging.Logger,
) -> MemoryService:
    service_key_raw = _derive_service_key(payload, extract_scope=extract_scope)

    client_profiles = payload.get("llm_profiles") if isinstance(payload.get("llm_profiles"), dict) else {}
    llm_profiles = _merge_llm_profiles(
        default_llm_profiles_from_server_config(config),
        client_profiles,
    )
    step_models_cfg = (config.get("llm", {}) if isinstance(config.get("llm"), dict) else {}).get("step_models", {})
    _validated_step_models(
        step_models_cfg,
        llm_profiles=llm_profiles,
        logger=logger,
    )
    step_temps = (config.get("llm") or {}).get("step_temperatures")
    if isinstance(step_temps, dict):
        merged_default = llm_profiles.get("default", {})
        for step_name, temp in step_temps.items():
            if temp is not None and temp != "":
                existing = llm_profiles.get(step_name, {**merged_default})
                existing["temperature"] = float(temp)
                llm_profiles[step_name] = existing
    payload["llm_profiles"] = llm_profiles
    database_config = payload.get("database_config")
    blob_config = payload.get("blob_config")

    if not isinstance(database_config, dict):
        scope_hint = extract_scope(payload) if isinstance(payload, dict) else None
        database_config = database_config_from_cfg(config, scope=scope_hint)
        payload["database_config"] = database_config

    if not isinstance(blob_config, dict):
        blob_config = blob_config_from_cfg(config)
        payload["blob_config"] = blob_config

    if isinstance(database_config, dict):
        ms = database_config.get("metadata_store")
        if isinstance(ms, dict) and str(ms.get("provider") or "").lower() == "sqlite":
            scope_hint2 = extract_scope(payload)
            soul_id2 = str((scope_hint2 or {}).get("soul_id") or "").strip()
            if not soul_id2:
                raise HTTPException(status_code=400, detail="soul_id required for sqlite scope")
            base = normalize_sqlite_dsn(str(ms.get("dsn") or ""))
            scope_for_dsn = dict(scope_hint2 or {})
            scope_for_dsn["soul_id"] = soul_id2
            ms["dsn"] = sqlite_dsn_for_scope(config, base, scope_for_dsn)

    blob_config = payload.get("blob_config") or {}
    memorize_config = payload.get("memorize_config") or {}

    fixed_cats = categories_from_cfg(config)
    if isinstance(memorize_config, dict):
        cats_cfg = (config.get("categories") or {}) if isinstance(config.get("categories"), dict) else {}
        if fixed_cats:
            memorize_config["memory_categories"] = fixed_cats
        memorize_config["dynamic_category_cluster_size"] = int(cats_cfg.get("dynamic_category_cluster_size", 3) or 3)
        memorize_config["max_categories_total"] = int((cats_cfg.get("max_total", 12)) or 0)
        memorize_config["episodes_per_segment"] = episodes_per_segment
        mem_cfg = config.get("memorize") if isinstance(config.get("memorize"), dict) else {}
        for passthrough_key in (
            "enable_confidence_normalization",
            "semantic_dedupe_enabled",
            "background_extra_messages_tokens",
        ):
            if passthrough_key in mem_cfg and passthrough_key not in memorize_config:
                memorize_config[passthrough_key] = mem_cfg[passthrough_key]
        if isinstance(step_models_cfg, dict):
            for cfg_key, profile_field in _STEP_MODEL_TO_MEMORIZE_PROFILE_FIELD.items():
                if profile_field not in memorize_config and str(step_models_cfg.get(cfg_key) or "").strip():
                    memorize_config[profile_field] = cfg_key
    retrieve_config = payload.get("retrieve_config")
    if not isinstance(retrieve_config, dict):
        retrieve_config = {}
        payload["retrieve_config"] = retrieve_config
    if isinstance(step_models_cfg, dict):
        for cfg_key, profile_field in _STEP_MODEL_TO_RETRIEVE_PROFILE_FIELD.items():
            if profile_field not in retrieve_config and str(step_models_cfg.get(cfg_key) or "").strip():
                retrieve_config[profile_field] = cfg_key
    user_config = payload.get("user_config") or {}

    sig = payload_signature(payload)
    service_key = f"{service_key_raw}__{sig}"
    storage_fp = _service_storage_fingerprint(
        database_config if isinstance(database_config, dict) else None,
        sqlite_file_from_dsn=sqlite_file_from_dsn,
        logger=logger,
    )

    with _SERVICES_LOCK:
        svc = _SERVICES.get(service_key)
        if svc is not None:
            prev_fp = _SERVICE_STORAGE_FP.get(service_key)
            if prev_fp == storage_fp:
                return svc
            _SERVICES.pop(service_key, None)
            _SERVICE_STORAGE_FP.pop(service_key, None)

        user_config = {**(user_config if isinstance(user_config, dict) else {}), "model": st_user_model}

        svc = MemoryService(
            llm_profiles=llm_profiles,
            blob_config=blob_config,
            database_config=database_config,
            memorize_config=memorize_config,
            retrieve_config=retrieve_config,
            user_config=user_config,
            claude_code=bool(config.get("claude_code", False)),
            claude_code_model=str(config.get("claude_code_model", "claude-opus-4-7")),
            claude_code_effort=str(config.get("claude_code_effort", "medium")),
            claude_code_permission_mode=str(config.get("claude_code_permission_mode", "")).strip() or None,
            claude_code_settings=str(config.get("claude_code_settings", "")).strip() or None,
            claude_code_workspace=str(config.get("claude_code_workspace", "")).strip() or None,
            claude_code_timeout_seconds=_cfg_int(config, "claude_code_timeout_seconds", 3600, minimum=1),
        )
        if log_prompts:
            svc.intercept_before_llm_call(prompt_log_before, name="prompt_logger")
            svc.intercept_after_llm_call(prompt_log_after, name="response_logger")
            svc.intercept_on_error_llm_call(prompt_log_on_error, name="error_logger")

        if len(_SERVICES) >= 50:
            oldest_key = next(iter(_SERVICES), None)
            if oldest_key is not None:
                _SERVICES.pop(oldest_key, None)
                _SERVICE_STORAGE_FP.pop(oldest_key, None)

        _SERVICES[service_key] = svc
        _SERVICE_STORAGE_FP[service_key] = storage_fp
        return svc

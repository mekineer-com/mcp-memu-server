import copy
import json
import os
import re
import tempfile
from pathlib import Path, PurePath
from typing import Any

STARTUP_WARNINGS: list[str] = []
STORAGE_STATUS: dict[str, Any] = {
    "ok": None,
    "provider": None,
    "dsn": None,
    "sqlite_path": None,
    "sqlite_parent": None,
    "sqlite_exists": None,
    "sqlite_open_ok": None,
    "sqlite_dir": None,
    "error": None,
}

EMBEDDING_PROVIDER = "gemini"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/"
EMBEDDING_PROFILE = f"{EMBEDDING_MODEL}:3072"
MODEL_STEPS = ("preprocess", "memory_extract", "category_update", "reflection", "consolidation")
REPLY_MODELS = ("srmm", "other_apps")
_MODEL_OPERATIONS = {
    "st_reply": (), "other_reply": (), "retrieve": ("reflection",),
    "rollup": ("preprocess",),
    "memorize": ("preprocess", "memory_extract", "category_update", "consolidation"),
    "consolidation": ("category_update", "consolidation"),
}


class SoulIdError(ValueError):
    pass


class SoulNameConflictError(SoulIdError):
    pass


_WINDOWS_RESERVED_FILENAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _set_storage_status(values: dict[str, Any]) -> None:
    STORAGE_STATUS.clear()
    STORAGE_STATUS.update(values)


def is_ephemeral_db(cfg: dict[str, Any]) -> bool:
    if not isinstance(cfg, dict):
        return False
    storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
    ms = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}
    provider = str(ms.get("provider") or "").lower()
    dsn = str(ms.get("dsn") or "")

    if provider == "sqlite" and ":memory:" in dsn:
        return True
    return False


def default_config() -> dict[str, Any]:
    server_dir = config_dir()
    memu_dir = server_dir.parent / "memu"
    memu_src = memu_dir / "src"

    sqlite_path = Path(":memory:")  # placeholder; per-soul dbs resolved per-request
    resources_dir = memu_dir / "resources"
    channels_data = server_dir.parent / "hermes-channels" / "data"

    return {
        "memu": {
            "path": str(memu_src) if memu_src.is_dir() else "",
        },
        "python": {
            "executable": "",
            "force_venv": False,
        },
        "pid_file": str(server_dir / ".memu-server.pid"),
        "listen": {"host": "127.0.0.1", "port": 8099},
        "llm": {
            "provider": "openai",
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "chat_model": "",
            "context_window_tokens": None,
            "step_context_window_tokens": {},
            "reply_models": {name: {"source": "config"} for name in REPLY_MODELS},
            "step_models": {name: {"source": "config"} for name in MODEL_STEPS},
            "embedding": {},
            "endpoint_overrides": {},
        },
        "storage": {
            "resources_dir": str(resources_dir),
            "metadata_store": {
                "provider": "sqlite",
                "dsn": "sqlite:///:memory:",
                "ddl_mode": "create",
            },
            "sqlite_dir": str(sqlite_path.parent),
        },
        "hermes": {
            "home": str(channels_data),
            "state_db_path": str(channels_data / "state.db"),
            "sessions_index_path": str(channels_data / "sessions" / "sessions.json"),
            "whatsapp_web_source_db": str(channels_data / "whatsapp" / "web_source.db"),
        },
        "mentra": {
            "enabled": False,
            "public_base_url": "",
            "gemini_api_key": "",
            "model": "gemini-2.5-flash-native-audio-preview-12-2025",
            "voice": "Kore",
            "session_warning_seconds": 0,
        },
        "categories": {
            "dynamic_category_cluster_size": 10,
            "category_summary_target_words": 300,
        },
        "retrieve": {
            "apimw_enabled": True,
            "apimw_cadence": 5,
            "apimw_memory_count": 20,
            "apimw_random_count": 5,
            "mental_health_query": True,
        },
        "consolidation_interval_days": 7,
        "claude_code": False,
        "claude_code_context_window_tokens": None,
        "procedural": {
            "yaml_dir": "../memu/procedural",
            "db_path": "../memu/sqlite/procedural.db",
        },
        "mcp": {"http_path": "/mcp", "sse_path": "/sse"},
    }


def config_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "config.json").resolve()


def config_dir() -> Path:
    return config_path().parent


def resolve_cfg_path(raw: str) -> Path:
    p = Path(str(raw or "")).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (config_dir() / p).resolve()


def sqlite_file_from_dsn(dsn: str) -> Path | None:
    """Parse a sqlite DSN into a filesystem path (minimal).

    Supports:
      sqlite:////abs/path.db
      sqlite:////abs/path.db?query
      sqlite:///abs-or-relative.db

    Returns None for :memory: or unknown formats.
    """
    if not isinstance(dsn, str):
        return None
    dsn = dsn.strip()
    if not dsn or ":memory:" in dsn:
        return None

    base = dsn.split("?", 1)[0]
    if base.startswith("sqlite:////"):
        return Path("/" + base[len("sqlite:////") :])
    if base.startswith("sqlite:///"):
        return Path(base[len("sqlite:///") :])
    return None


def sqlite_dsn_from_path(path: PurePath) -> str:
    return f"sqlite:///{path.as_posix()}"


def normalize_sqlite_dsn(dsn_or_path: str) -> str:
    """Accept either a sqlite DSN or a plain filesystem path.

    Users frequently paste a path like "/home/.../memu.db" instead of a DSN.
    This converts it into a correct absolute sqlite DSN.
    """
    raw = str(dsn_or_path or "").strip()
    if not raw:
        return raw
    if raw == ":memory:" or raw.lower() == "memory":
        return "sqlite:///:memory:"
    if raw.startswith("sqlite:"):
        f = sqlite_file_from_dsn(raw)
        if f is None:
            return raw
        p = f.expanduser()
        if not p.is_absolute():
            p = (config_dir() / p).resolve()
        else:
            p = p.resolve()
        return sqlite_dsn_from_path(p)

    p = resolve_cfg_path(raw)
    return sqlite_dsn_from_path(p)


def sqlite_dir_from_cfg(cfg: dict[str, Any], fallback_dsn: str | None = None) -> Path:
    storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
    d = storage.get("sqlite_dir")
    if isinstance(d, str) and d.strip():
        return resolve_cfg_path(d)

    if fallback_dsn:
        f = sqlite_file_from_dsn(str(fallback_dsn))
        if f is not None:
            return f.expanduser().resolve().parent

    return (config_dir() / "sqlite").resolve()


def ensure_storage_paths(cfg: dict[str, Any]) -> None:
    try:
        storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}

        resources_dir = storage.get("resources_dir")
        if isinstance(resources_dir, str) and resources_dir.strip():
            resolve_cfg_path(resources_dir).mkdir(parents=True, exist_ok=True)

        ms = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}
        provider = str(ms.get("provider") or "").lower()
        dsn = str(ms.get("dsn") or "")

        _set_storage_status(
            {
                "ok": True,
                "provider": provider or None,
                "dsn": dsn or None,
                "sqlite_path": None,
                "sqlite_parent": None,
                "sqlite_exists": None,
                "sqlite_open_ok": None,
                "sqlite_dir": str(sqlite_dir_from_cfg(cfg, dsn)),
                "error": None,
            }
        )

        if provider == "sqlite":
            dsn = normalize_sqlite_dsn(dsn)
            STORAGE_STATUS["dsn"] = dsn or None
            ms["dsn"] = dsn

            sqlite_dir = sqlite_dir_from_cfg(cfg, dsn)
            sqlite_dir.mkdir(parents=True, exist_ok=True)
            try:
                test_path = sqlite_dir / ".write_test"
                test_path.write_text("ok", encoding="utf-8")
                test_path.unlink(missing_ok=True)
            except OSError as e:
                STORAGE_STATUS["sqlite_open_ok"] = False
                STORAGE_STATUS["ok"] = False
                STORAGE_STATUS["error"] = f"sqlite_dir_write: {type(e).__name__}: {e}"
                STARTUP_WARNINGS.append(STORAGE_STATUS["error"])
    except OSError as e:
        STORAGE_STATUS["ok"] = False
        STORAGE_STATUS["error"] = f"storage_paths: {type(e).__name__}: {e}"
        STARTUP_WARNINGS.append(STORAGE_STATUS["error"])


def load_config() -> dict[str, Any]:
    p = config_path()
    if not p.exists():
        cfg = normalize_model_mappings(default_config())
        save_config(cfg)
        ensure_storage_paths(cfg)
        return cfg
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a JSON object: {p}")
    cfg = normalize_model_mappings(cfg)
    ensure_storage_paths(cfg)
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=p.parent,
            prefix=f".{p.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            json.dump(cfg, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, p)
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def mask_config(cfg: dict[str, Any], *, include_llm_secret: bool = False) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    def mask(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "api_key" and child:
                    value[key] = "***"
                else:
                    mask(child)
        elif isinstance(value, list):
            for child in value:
                mask(child)
    if not include_llm_secret:
        llm = out.get("llm")
        base_key = llm.get("api_key") if isinstance(llm, dict) else None
        mask(llm)
        if isinstance(base_key, str) and base_key:
            llm["api_key"] = base_key[:4] + "..." + base_key[-4:]
    mentra = out.get("mentra")
    if isinstance(mentra, dict) and mentra.get("gemini_api_key"):
        mentra["gemini_api_key"] = "***"
    return out


def get_storage_dir(cfg: dict[str, Any]) -> Path:
    d = resolve_cfg_path(str(cfg.get("storage", {}).get("resources_dir") or "./storage"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def procedural_yaml_dir(cfg: dict[str, Any]) -> Path:
    raw = str((cfg.get("procedural") or {}).get("yaml_dir") or "../memu/procedural")
    return resolve_cfg_path(raw)


def procedural_db_path(cfg: dict[str, Any]) -> Path:
    raw = str((cfg.get("procedural") or {}).get("db_path") or "../memu/sqlite/procedural.db")
    return resolve_cfg_path(raw)


def procedural_should_ingest(db_path: Path, yaml_dir: Path) -> bool:
    if not yaml_dir.exists():
        return False
    if not db_path.exists():
        return True
    db_mtime = db_path.stat().st_mtime
    return any(yf.stat().st_mtime > db_mtime for yf in yaml_dir.glob("*.yaml"))


def sanitize_db_filename(name: str) -> str:
    s = str(name or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    if not s:
        s = "unknown"
    return s[:80]


def validate_soul_id(name: Any) -> str:
    soul_id = str(name or "").strip()
    if (
        not soul_id
        or soul_id in {".", ".."}
        or len(soul_id.encode("utf-8")) > 80
        or soul_id.endswith(".")
        or soul_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED_FILENAMES
        or any(ord(c) < 32 or ord(c) == 127 or c in '/\\*?[]:"<>|' for c in soul_id)
    ):
        raise SoulIdError("Invalid soul name")
    return soul_id


def soul_gen_config_path(cfg: dict[str, Any], user_id: str, soul_id: str) -> Path:
    user_part = sanitize_db_filename(user_id)
    soul_part = validate_soul_id(soul_id)
    return sqlite_dir_from_cfg(cfg) / f"{user_part}__{soul_part}.gen.json"


def load_soul_gen_config(cfg: dict[str, Any], user_id: str, soul_id: str, logger: Any | None = None) -> dict[str, Any]:
    p = soul_gen_config_path(cfg, user_id, soul_id)
    if not p.exists():
        return {}
    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        if logger is not None:
            logger.warning("Ignoring invalid soul generation config at %s", p, exc_info=True)
        return {}
    if not isinstance(parsed, dict):
        if logger is not None:
            logger.warning("Ignoring non-object soul generation config at %s (type=%s)", p, type(parsed).__name__)
        return {}
    return parsed


def sqlite_path_for_scope(
    cfg: dict[str, Any], base_dsn: str, scope: dict[str, Any] | None
) -> Path | None:
    if not isinstance(scope, dict):
        scope = {}
    soul_id = str(scope.get("soul_id") or "").strip()
    if not soul_id:
        return None
    db_path = sqlite_dir_from_cfg(cfg, fallback_dsn=base_dsn) / f"{validate_soul_id(soul_id)}.db"
    try:
        conflict = next(
            (
                path for path in db_path.parent.iterdir()
                if path.name.casefold() == db_path.name.casefold() and path.name != db_path.name
            ),
            None,
        )
    except FileNotFoundError:
        conflict = None
    if conflict is not None:
        raise SoulNameConflictError(f"Soul name is already taken as {conflict.stem!r}")
    return db_path


def sqlite_dsn_for_scope(cfg: dict[str, Any], base_dsn: str, scope: dict[str, Any] | None) -> str:
    if not isinstance(scope, dict) or not scope.get("soul_id"):
        return base_dsn
    from app.services.owner import require_owner

    require_owner(cfg, scope.get("user_id"))
    db_path = sqlite_path_for_scope(cfg, base_dsn, scope)
    if db_path is None:
        return base_dsn
    configured_dsn = str((((cfg.get("storage") or {}).get("metadata_store") or {}).get("dsn") or ""))
    configured_path = sqlite_file_from_dsn(normalize_sqlite_dsn(configured_dsn)) if configured_dsn else None
    if configured_path is not None and db_path.resolve() == configured_path.resolve():
        raise SoulIdError("Soul name is reserved by the base database")
    if db_path.resolve() == procedural_db_path(cfg).resolve():
        raise SoulIdError("Soul name is reserved by the procedural database")
    if db_path.is_symlink():
        raise SoulIdError("Soul database must not be a symlink")
    if not db_path.exists():
        raise SoulIdError(f"Soul database does not exist: {db_path.stem!r}")
    if not db_path.is_file():
        raise SoulIdError("Soul database must be a file")
    return sqlite_dsn_from_path(db_path)


def database_config_from_cfg(cfg: dict[str, Any], scope: dict[str, Any] | None = None) -> dict[str, Any]:
    storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
    meta = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}

    provider = meta.get("provider") or "sqlite"
    provider = str(provider).strip().lower() or "sqlite"
    if provider != "sqlite":
        raise RuntimeError(f"Unsupported storage.metadata_store.provider={provider!r}; only 'sqlite' is supported.")

    dsn = meta.get("dsn")
    if not dsn:
        dsn = default_config()["storage"]["metadata_store"]["dsn"]

    dsn = normalize_sqlite_dsn(str(dsn))
    dsn = sqlite_dsn_for_scope(cfg, dsn, scope or {})

    ddl_mode = meta.get("ddl_mode") or "create"

    metadata_store = {"provider": provider, "dsn": dsn, "ddl_mode": ddl_mode}
    metadata_store["embedding_profile"] = EMBEDDING_PROFILE
    return {"metadata_store": metadata_store}


def blob_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"resources_dir": str(get_storage_dir(cfg))}


def _base_chat_profile(llm: dict[str, Any]) -> dict[str, Any]:
    api_key = str(llm.get("api_key") or "")
    base_url = str(llm.get("base_url") or "https://api.openai.com/v1")
    chat_model = str(llm.get("chat_model") or "")
    provider = str(llm.get("provider") or "openai")
    endpoint_overrides = llm.get("endpoint_overrides") or {}
    temperature = llm.get("temperature")
    default_profile: dict[str, Any] = {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "chat_model": chat_model,
        "endpoint_overrides": endpoint_overrides,
    }
    if temperature is not None:
        default_profile["temperature"] = float(temperature)
    max_tokens = llm.get("max_tokens")
    if max_tokens is not None:
        default_profile["max_tokens"] = max_tokens
    if llm.get("context_window_tokens") is not None:
        default_profile["context_window_tokens"] = llm["context_window_tokens"]
    return default_profile


def normalize_model_mappings(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    llm = out.setdefault("llm", {})
    if not isinstance(llm, dict):
        raise ValueError("llm must be an object")
    capacities = llm.get("step_context_window_tokens") or {}
    if not isinstance(capacities, dict):
        raise ValueError("llm.step_context_window_tokens must be an object")
    for field, names in (("reply_models", REPLY_MODELS), ("step_models", MODEL_STEPS)):
        rows = llm.setdefault(field, {})
        if not isinstance(rows, dict):
            raise ValueError(f"llm.{field} must be an object")
        for name in names:
            row = rows.get(name)
            if row is None or row == "":
                row = {"source": "config"}
            elif isinstance(row, str) and field == "step_models":
                profile = _base_chat_profile(llm)
                model = row.strip()
                if model != profile["chat_model"]:
                    profile.pop("context_window_tokens", None)
                profile["chat_model"] = model
                if capacities.get(name) is not None:
                    profile["context_window_tokens"] = capacities[name]
                row = {"source": "configured", "selection": {"provider": profile["provider"], "model": model},
                       "resolved": profile} if model else {"source": "config"}
            if not isinstance(row, dict) or row.get("source") not in {"config", "srmm", "st_native", "configured"}:
                raise ValueError(f"Invalid model selector: llm.{field}.{name}")
            if row["source"] == "srmm" and field != "step_models":
                raise ValueError("srmm inheritance is only allowed for advanced steps")
            for key in ("selection", "resolved"):
                if key in row and not isinstance(row[key], dict):
                    raise ValueError(f"llm.{field}.{name}.{key} must be an object")
            if "resolved" in row and row["resolved"].get("provider") != "openai":
                raise ValueError(f"llm.{field}.{name}.resolved must use an OpenAI-compatible provider")
            rows[name] = row
    if "st_selected" in llm and not isinstance(llm["st_selected"], dict):
        raise ValueError("llm.st_selected must be an object")
    snapshot = llm.get("st_selected", {})
    if "resolved" in snapshot and (not isinstance(snapshot["resolved"], dict) or snapshot["resolved"].get("provider") != "openai"):
        raise ValueError("llm.st_selected.resolved must use an OpenAI-compatible provider")
    return out


def merge_model_config(current: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("Config must be an object")
    def reject_masks(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "api_key" and isinstance(child, str) and (child == "***" or re.fullmatch(r".{4}\.\.\..{4}", child)):
                    raise ValueError("Masked api_key cannot be saved")
                reject_masks(child)
        elif isinstance(value, list):
            for child in value:
                reject_masks(child)
    reject_masks(body.get("llm"))
    merged = {**copy.deepcopy(current), **copy.deepcopy(body)}
    for field in ("llm", "mentra"):
        if field not in body:
            continue
        if not isinstance(body[field], dict):
            raise ValueError(f"{field} must be an object")
        merged[field] = {**copy.deepcopy(current.get(field) or {}), **copy.deepcopy(body[field])}
    incoming_storage = body.get("storage")
    if isinstance(incoming_storage, dict):
        storage = current.get("storage") if isinstance(current.get("storage"), dict) else {}
        merged["storage"] = {**copy.deepcopy(storage), **copy.deepcopy(incoming_storage)}
        incoming_meta = incoming_storage.get("metadata_store")
        if isinstance(incoming_meta, dict):
            meta = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}
            merged["storage"]["metadata_store"] = {**copy.deepcopy(meta), **copy.deepcopy(incoming_meta)}
    incoming = body.get("llm", {})
    for field in ("reply_models", "step_models", "step_context_window_tokens", "step_temperatures", "embedding"):
        if field in incoming:
            if not isinstance(incoming[field], dict):
                raise ValueError(f"llm.{field} must be an object")
            merged["llm"][field] = {**copy.deepcopy((current.get("llm") or {}).get(field) or {}), **copy.deepcopy(incoming[field])}
    if "claude_code" in body and type(body["claude_code"]) is not bool:
        raise ValueError("claude_code must be a Boolean")
    return normalize_model_mappings(merged)


def _resolve_model_profile(llm: dict[str, Any], name: str, *, validate: bool = True) -> dict[str, Any]:
    field = "reply_models" if name in REPLY_MODELS else "step_models"
    row = llm[field][name]
    source = row["source"]
    if source == "srmm":
        return _resolve_model_profile(llm, "srmm", validate=validate)
    if source == "config":
        profile = _base_chat_profile(llm)
        capacity = (llm.get("step_context_window_tokens") or {}).get(name)
        if capacity is not None:
            profile["context_window_tokens"] = capacity
    else:
        holder = llm.get("st_selected", {}) if source == "st_native" else row
        profile = copy.deepcopy(holder.get("resolved"))
        if not isinstance(profile, dict):
            raise ValueError(f"Model mapping {name} needs a resolved {source} profile")
    if validate or source != "config":
        if profile.get("provider") != "openai":
            raise ValueError(f"Model mapping {name} requires an OpenAI-compatible profile")
        if not isinstance(profile.get("base_url"), str) or not profile["base_url"].strip():
            raise ValueError(f"Model mapping {name} needs base_url")
        if not isinstance(profile.get("chat_model"), str) or not profile["chat_model"].strip():
            raise ValueError(f"Model mapping {name} needs chat_model")
        if "api_key" not in profile or not isinstance(profile["api_key"], str):
            raise ValueError(f"Model mapping {name} needs an explicit api_key (empty for a keyless endpoint)")
    for key in ("context_window_tokens", "max_tokens"):
        value = profile.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"Model mapping {name}.{key} must be a positive integer")
    return profile


def default_llm_profiles_from_server_config(cfg: dict[str, Any], *, operation: str | None = None) -> dict[str, Any]:
    if operation is not None and operation not in _MODEL_OPERATIONS:
        raise ValueError(f"Unknown model operation: {operation}")
    normalized = normalize_model_mappings(cfg)
    llm = normalized["llm"]
    default_profile = _base_chat_profile(llm)
    embedding_cfg = llm.get("embedding") if isinstance(llm.get("embedding"), dict) else {}
    embedding_profile = {
        "provider": EMBEDDING_PROVIDER,
        "api_key": str(embedding_cfg.get("api_key") or ""),
        "base_url": str(embedding_cfg.get("base_url") or EMBEDDING_BASE_URL),
        "embed_model": EMBEDDING_MODEL,
        "endpoint_overrides": embedding_cfg.get("endpoint_overrides") or {},
    }
    profiles = {
        "default": default_profile,
        "embedding": embedding_profile,
    }
    if operation != "st_reply" and cfg.get("claude_code") is True:
        return profiles
    if operation in {"st_reply", "other_reply"}:
        profiles["default"] = _resolve_model_profile(llm, "srmm" if operation == "st_reply" else "other_apps")
    else:
        names = _MODEL_OPERATIONS[operation] if operation is not None else MODEL_STEPS
        for name in names:
            profiles[name] = _resolve_model_profile(llm, name, validate=operation is not None)
    return profiles

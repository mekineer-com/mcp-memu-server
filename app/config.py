import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

from app.db import sqlite_ensure_nonempty as _sqlite_ensure_nonempty


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


def home_dir() -> Path:
    h = os.getenv("HOME") or os.getenv("USERPROFILE") or "."
    return Path(h).expanduser().resolve()


def default_config() -> dict[str, Any]:
    home = home_dir()

    memu_guess = None
    for cand in (home / "apps" / "memu",):
        if cand.exists():
            memu_guess = cand
            break

    sqlite_path = Path(":memory:")  # placeholder; per-soul dbs resolved per-request
    resources_dir = home / "apps" / "memu" / "resources"

    return {
        "memu": {
            "path": str(memu_guess) if memu_guess else "",
        },
        "python": {
            "executable": "",
            "force_venv": False,
        },
        "pid_file": str(home / "apps" / "mcp-memu-server" / ".memu-server.pid"),
        "listen": {"host": "127.0.0.1", "port": 8099},
        "llm": {
            "provider": "openai",
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "chat_model": "",
            "embed_model": "",
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
            "home": str(home / ".hermes"),
            "state_db_path": str(home / ".hermes" / "state.db"),
            "sessions_index_path": str(home / ".hermes" / "sessions" / "sessions.json"),
            "whatsapp_history_source": "web_source",
            "whatsapp_web_source_db": str(home / ".hermes" / "whatsapp" / "web_source.db"),
            "whatsapp_history_limit": 250,
            "whatsapp_reply_prefix": "",
        },
        "categories": {
            "defaults": ["personal_info", "preferences", "relationships", "goals"],
            "max_total": 12,
            "dynamic_category_cluster_size": 3,
        },
        "retrieve": {
            "apimw_enabled": True,
            "apimw_cadence": 5,
            "apimw_memory_count": 20,
            "apimw_random_count": 5,
            "mental_health_query": True,
        },
        "consolidation_interval_days": 7,
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
        return f"sqlite:////{p.as_posix().lstrip('/')}"

    p = resolve_cfg_path(raw)
    return f"sqlite:////{p.as_posix().lstrip('/')}"


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
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg = default_config()
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        ensure_storage_paths(cfg)
        return cfg
    try:
        raw = p.read_text(encoding="utf-8")
        cfg = json.loads(raw) if raw.strip() else default_config()
        if not isinstance(cfg, dict):
            cfg = default_config()
        ensure_storage_paths(cfg)
        return cfg
    except (json.JSONDecodeError, OSError):
        traceback.print_exc()
        cfg = default_config()
        ensure_storage_paths(cfg)
        return cfg


def save_config(cfg: dict[str, Any]) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def mask_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    key = out.get("llm", {}).get("api_key", "")
    if isinstance(key, str) and key:
        out["llm"]["api_key"] = key[:4] + "..." + key[-4:]
    return out


def categories_from_cfg(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cats_cfg = cfg.get("categories") if isinstance(cfg.get("categories"), dict) else {}
    raw = cats_cfg.get("defaults") if isinstance(cats_cfg.get("defaults"), list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for c in raw:
        name = None
        desc = ""
        if isinstance(c, str):
            name = c.strip()
        elif isinstance(c, dict):
            name = str(c.get("name") or "").strip()
            desc = str(c.get("description") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "description": desc})
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


def soul_gen_config_path(cfg: dict[str, Any], user_id: str, soul_id: str) -> Path:
    user_part = sanitize_db_filename(user_id)
    soul_part = sanitize_db_filename(soul_id)
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


def save_soul_gen_config(cfg: dict[str, Any], user_id: str, soul_id: str, data: dict[str, Any]) -> None:
    soul_gen_config_path(cfg, user_id, soul_id).write_text(json.dumps(data, indent=2))


def sqlite_dsn_for_scope(cfg: dict[str, Any], base_dsn: str, scope: dict[str, Any] | None) -> str:
    if not isinstance(scope, dict):
        scope = {}

    soul_id = str(scope.get("soul_id") or "").strip()

    sqlite_dir = sqlite_dir_from_cfg(cfg, fallback_dsn=base_dsn)
    sqlite_dir.mkdir(parents=True, exist_ok=True)

    if not soul_id:
        return base_dsn

    basename = sanitize_db_filename(soul_id)
    db_path = (sqlite_dir / f"{basename}.db").resolve()
    _sqlite_ensure_nonempty(db_path)
    return f"sqlite:////{db_path.as_posix().lstrip('/')}"


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

    return {
        "metadata_store": {
            "provider": provider,
            "dsn": dsn,
            "ddl_mode": ddl_mode,
        }
    }


def blob_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"resources_dir": str(get_storage_dir(cfg))}


def default_llm_profiles_from_server_config(cfg: dict[str, Any]) -> dict[str, Any]:
    llm = cfg.get("llm", {}) if isinstance(cfg.get("llm"), dict) else {}
    api_key = str(llm.get("api_key") or "")
    base_url = str(llm.get("base_url") or "https://api.openai.com/v1")
    chat_model = str(llm.get("chat_model") or "")
    embed_model = str(llm.get("embed_model") or "")
    provider = str(llm.get("provider") or "openai")
    endpoint_overrides = llm.get("endpoint_overrides") or {}
    temperature = llm.get("temperature")
    default_profile: dict[str, Any] = {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "chat_model": chat_model,
        "embed_model": embed_model,
        "endpoint_overrides": endpoint_overrides,
    }
    if temperature is not None:
        default_profile["temperature"] = float(temperature)
    max_tokens = llm.get("max_tokens")
    if max_tokens is not None:
        default_profile["max_tokens"] = int(max_tokens)
    profiles = {
        "default": default_profile,
        "embedding": {**default_profile},
    }
    step_models = llm.get("step_models")
    if isinstance(step_models, dict):
        for step_name, model in step_models.items():
            model_str = str(model or "").strip()
            if model_str:
                profiles[step_name] = {**default_profile, "chat_model": model_str}
    return profiles

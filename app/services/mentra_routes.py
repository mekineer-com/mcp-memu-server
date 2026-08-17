from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException


def register_mentra_routes(app: FastAPI, *, get_config: Callable[[], dict[str, Any]]) -> None:
    async def require_bearer(authorization: str | None = Header(default=None)) -> None:
        config = get_config().get("mentra") or {}
        if not config.get("enabled"):
            raise HTTPException(status_code=404, detail="Not Found")

        expected = str(config.get("integration_bearer_token") or "")
        if not expected:
            raise HTTPException(status_code=503, detail="Mentra bearer credential is not configured")

        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            supplied.encode(), expected.encode()
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid Mentra bearer credential",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get(
        "/integration/mentra/health",
        tags=["integration"],
        dependencies=[Depends(require_bearer)],
    )
    async def mentra_health() -> dict[str, bool]:
        return {"ok": True}

from pydantic import BaseModel

from app import main
from memu.app.service import MemoryService


class _TestScope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


class SavedBatchService:
    """Fake extraction, real categorize/dedupe transactions and publication hooks."""

    @staticmethod
    def make_engine(scope):
        path = main._sqlite_current_path(scope["user_id"], scope["soul_id"])
        main._sqlite_ensure_nonempty(path)
        with main._sqlite_connect(path) as con:
            main._sqlite_ensure_conversation_state_schema(con)
        return MemoryService(
            database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{path}"}},
            user_config={"model": _TestScope},
        )

    async def memorize_segments_batch(self, **kwargs):
        scope = {key: kwargs["user"][key] for key in ("user_id", "soul_id")}
        self.engine = self.make_engine(scope)
        self.engine._select_embedding_client = lambda *_a, **_kw: object()
        results = await self.extract(**kwargs)
        for segment, result in zip(kwargs["segments"], results, strict=True):
            plan = {**segment["segment"], "resource_url": segment["resource_url"], "entries": []}
            url = segment["resource_url"]
            state = {"ctx": self.engine._get_context(), "store": self.engine.database,
                     "user": scope, "modality": "conversation", "local_path": segment["local_path"],
                     "conversation_id": kwargs["conversation_id"], "segment_plans": [plan],
                     "on_segment_saved": kwargs.get("on_segment_saved"),
                     "on_dedupe_complete": (lambda session, url=url: kwargs["on_dedupe_complete"](session, url))
                     if not plan.get("context_only") else None}
            await self.engine._memorize_categorize_items(state, None)
            await self.engine._memorize_dedupe_merge(state, None)
            await self.engine._memorize_persist_and_index(state, None)
            result["pending_segment_ids"] = state["pending_segment_ids"]
        return results

    @property
    def database(self):
        return self.engine.database

    async def resume_memorize_segment(self, **kwargs):
        self.engine = self.make_engine(kwargs["user"])
        await self.engine.resume_memorize_segment(**kwargs)

    def _sqlite_write_session(self, store):
        return self.engine._sqlite_write_session(store)

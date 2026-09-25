from __future__ import annotations

import sqlite3

from app.db import sqlite_ensure_conversation_state_schema
from app.services import activity_messages


def test_activity_tail_honors_latest_memorize_display_floor() -> None:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    sqlite_ensure_conversation_state_schema(con)
    activity_messages.ensure_activity_messages_schema(con)
    con.execute(
        "INSERT INTO conversations ("
        "conversation_id, soul_id, user_id, memorize_chat, digest_cursor, "
        "last_memorize_at, "
        "last_display_segment_start_index, last_display_segment_end_index, "
        "last_display_segment_at"
        ") VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)",
        (
            "activity:dm:Fictional Soul",
            "Fictional Soul",
            "Fictional User",
            10,
            "2026-09-25T00:00:00+00:00",
            9,
            10,
            "2026-09-25T00:00:00+00:00",
        ),
    )
    con.executemany(
        "INSERT INTO activity_messages ("
        "user_id, soul_id, conversation_id, speaker, content, received_at"
        ") VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "Fictional User",
                "Fictional Soul",
                "activity:dm:Atomic",
                "Fictional Soul",
                f"Fictional recap {index}",
                f"2026-09-25T00:00:{index:02d}+00:00",
            )
            for index in range(1, 11)
        ],
    )
    con.commit()

    rows = activity_messages.load_activity_tail_for_ai(
        con,
        user_id="Fictional User",
        soul_id="Fictional Soul",
        recent_fallback_messages=8,
    )
    assert [row["content"] for row in rows] == [
        f"Fictional recap {index}" for index in range(3, 11)
    ]

    con.execute(
        "UPDATE conversations SET last_display_segment_start_index = NULL, "
        "last_display_segment_end_index = NULL, last_display_segment_at = NULL "
        "WHERE conversation_id = ?",
        ("activity:dm:Fictional Soul",),
    )
    con.commit()
    assert activity_messages.load_activity_tail_for_ai(
        con,
        user_id="Fictional User",
        soul_id="Fictional Soul",
        recent_fallback_messages=8,
    ) == []
    con.close()

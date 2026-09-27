from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


class VisitorTracker:
    """SQLite-backed visitor tracking for live totals and online-user counts."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visitor_sessions (
                    session_id TEXT PRIMARY KEY,
                    first_visit TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    ip_address TEXT,
                    total_page_views INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    page TEXT NOT NULL,
                    visited_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_visitor_sessions_last_seen ON visitor_sessions(last_seen)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_visit_log_session_id ON visit_log(session_id)"
            )
            conn.commit()

    def _cleanup_stale_sessions(self, idle_minutes: int = 5) -> None:
        cutoff = (datetime.utcnow() - timedelta(minutes=idle_minutes)).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute("DELETE FROM visitor_sessions WHERE last_seen < ?", (cutoff,))
            conn.commit()

    def track_visit(self, session_id: str, ip_address: str, page: str = "/") -> dict:
        now = datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO visitor_sessions (session_id, first_visit, last_seen, ip_address, total_page_views)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    ip_address = excluded.ip_address,
                    total_page_views = visitor_sessions.total_page_views + 1
                """,
                (session_id, now, now, ip_address),
            )
            conn.execute(
                "INSERT INTO visit_log (session_id, page, visited_at) VALUES (?, ?, ?)",
                (session_id, page, now),
            )
            conn.commit()

        return self.get_stats()

    def get_stats(self) -> dict:
        self._cleanup_stale_sessions()
        cutoff = (datetime.utcnow() - timedelta(minutes=5)).isoformat(timespec="seconds")
        with self._connect() as conn:
            total_visits = conn.execute("SELECT COUNT(*) AS c FROM visit_log").fetchone()["c"]
            unique_visitors = conn.execute(
                "SELECT COUNT(DISTINCT session_id) AS c FROM visitor_sessions"
            ).fetchone()["c"]
            online_now = conn.execute(
                "SELECT COUNT(*) AS c FROM visitor_sessions WHERE last_seen > ?",
                (cutoff,),
            ).fetchone()["c"]

        return {
            "total_visits": int(total_visits or 0),
            "unique_visitors": int(unique_visitors or 0),
            "online_now": int(online_now or 0),
        }


visitor_tracker = VisitorTracker(Path(__file__).resolve().parents[2] / "data" / "visitors.db")

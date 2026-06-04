"""参加者ロスター（名簿）管理。

大量動画を batch-process した後、各 SPEAKER_X を人物名に紐付ける際、
ロスターから補完候補を出すための簡易データベース。
正規参加者一覧 + 各人のメモ（部署・役職など）を保持する。
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


def _default_roster_path() -> Path:
    return Path(os.environ.get("ROSTER_DB_PATH", "./roster.sqlite3"))


@dataclass
class Member:
    name: str
    note: str = ""


class RosterDB:
    """参加者名簿（SQLite）。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else _default_roster_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS members (
                name TEXT PRIMARY KEY,
                note TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def add(self, name: str, note: str = "") -> Member:
        self._conn.execute(
            "INSERT OR REPLACE INTO members (name, note) VALUES (?, ?)",
            (name, note),
        )
        self._conn.commit()
        return Member(name=name, note=note)

    def remove(self, name: str) -> bool:
        cur = self._conn.execute("DELETE FROM members WHERE name = ?", (name,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_all(self) -> list[Member]:
        cur = self._conn.execute("SELECT name, note FROM members ORDER BY name")
        return [Member(name=r[0], note=r[1] or "") for r in cur.fetchall()]

    def exists(self, name: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM members WHERE name = ?", (name,))
        return cur.fetchone() is not None

    def close(self) -> None:
        self._conn.close()

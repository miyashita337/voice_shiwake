"""参加者ロスター単体テスト。"""

from __future__ import annotations

from click.testing import CliRunner

from voice_shiwake.cli import main
from voice_shiwake.roster import RosterDB


class TestRosterDB:
    def test_add_and_list(self, tmp_path):
        db = RosterDB(db_path=tmp_path / "r.sqlite3")
        try:
            db.add("山田", note="リード")
            db.add("鈴木")
            members = db.list_all()
            assert len(members) == 2
            names = [m.name for m in members]
            assert "山田" in names
            assert "鈴木" in names
        finally:
            db.close()

    def test_exists(self, tmp_path):
        db = RosterDB(db_path=tmp_path / "r.sqlite3")
        try:
            db.add("田中")
            assert db.exists("田中") is True
            assert db.exists("存在しない") is False
        finally:
            db.close()

    def test_remove(self, tmp_path):
        db = RosterDB(db_path=tmp_path / "r.sqlite3")
        try:
            db.add("佐藤")
            assert db.remove("佐藤") is True
            assert db.remove("佐藤") is False
        finally:
            db.close()

    def test_add_overwrites_note(self, tmp_path):
        db = RosterDB(db_path=tmp_path / "r.sqlite3")
        try:
            db.add("高橋", note="old")
            db.add("高橋", note="new")
            members = db.list_all()
            assert len(members) == 1
            assert members[0].note == "new"
        finally:
            db.close()


class TestRosterCLI:
    def test_roster_add_multiple(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROSTER_DB_PATH", str(tmp_path / "r.sqlite3"))
        runner = CliRunner()
        result = runner.invoke(main, ["roster", "add", "山田", "鈴木", "田中"])
        assert result.exit_code == 0
        list_result = runner.invoke(main, ["roster", "list"])
        assert "山田" in list_result.output
        assert "鈴木" in list_result.output
        assert "田中" in list_result.output
        assert "計 3 名" in list_result.output

    def test_roster_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROSTER_DB_PATH", str(tmp_path / "r.sqlite3"))
        runner = CliRunner()
        result = runner.invoke(main, ["roster", "list"])
        assert "ロスター未登録" in result.output

    def test_roster_remove_missing_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROSTER_DB_PATH", str(tmp_path / "r.sqlite3"))
        runner = CliRunner()
        result = runner.invoke(main, ["roster", "remove", "存在しない"])
        assert result.exit_code == 1

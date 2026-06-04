"""CLI 統合テスト（mock ベース）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from click.testing import CliRunner

from voice_shiwake.cli import main


class TestCLIList:
    def test_empty_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "empty.sqlite3"))
        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "登録済み声紋なし" in result.output


class TestCLIEnroll:
    def test_enroll_mocks_embedding(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "db.sqlite3"))
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([0.1, 0.2]))

        sample = tmp_path / "sample.wav"
        sample.touch()

        runner = CliRunner()
        result = runner.invoke(main, ["enroll", "--name", "田中", "--audio", str(sample)])
        assert result.exit_code == 0, result.output
        assert "登録完了" in result.output
        assert "田中" in result.output


class TestCLIDelete:
    def test_delete_missing_returns_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "db.sqlite3"))
        runner = CliRunner()
        result = runner.invoke(main, ["delete", "--name", "存在しない"])
        assert result.exit_code == 1

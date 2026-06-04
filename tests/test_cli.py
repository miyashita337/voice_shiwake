"""CLI 統合テスト（mock ベース）。"""

from __future__ import annotations

from pathlib import Path

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
        assert "累計 1" in result.output

    def test_enroll_appends_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "db.sqlite3"))
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([0.1, 0.2]))

        sample = tmp_path / "sample.wav"
        sample.touch()
        runner = CliRunner()
        # 2回 enroll
        runner.invoke(main, ["enroll", "--name", "田中", "--audio", str(sample)])
        result = runner.invoke(main, ["enroll", "--name", "田中", "--audio", str(sample)])
        assert "累計 2" in result.output

    def test_enroll_replace_resets_samples(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "db.sqlite3"))
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([0.1, 0.2]))

        sample = tmp_path / "sample.wav"
        sample.touch()
        runner = CliRunner()
        runner.invoke(main, ["enroll", "--name", "田中", "--audio", str(sample)])
        runner.invoke(main, ["enroll", "--name", "田中", "--audio", str(sample)])
        result = runner.invoke(main, ["enroll", "--name", "田中", "--audio", str(sample), "--replace"])
        assert "累計 1" in result.output
        assert "上書き" in result.output


class TestCLIDelete:
    def test_delete_missing_returns_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "db.sqlite3"))
        runner = CliRunner()
        result = runner.invoke(main, ["delete", "--name", "存在しない"])
        assert result.exit_code == 1


class TestCLICorrect:
    def test_correct_registers_mappings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "db.sqlite3"))
        from voice_shiwake import voiceprint as vp_mod

        # 異なる音声に対して異なる embedding を返すように
        embs = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "C": np.array([0.5, 0.5]),
        }
        def fake_compute(p: Path):
            for key, v in embs.items():
                if f"SPEAKER_{key}.wav" in str(p):
                    return v
            return np.array([0.0, 0.0])
        monkeypatch.setattr(vp_mod, "compute_embedding", fake_compute)

        # ディレクトリ＋ 3 個の wav を準備
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()
        for key in ["A", "B", "C"]:
            (samples_dir / f"SPEAKER_{key}.wav").touch()

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "correct",
                "--samples-dir", str(samples_dir),
                "--map", "A=渡辺",
                "--map", "B=小川",
                "--map", "C=林",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "渡辺" in result.output
        assert "小川" in result.output
        assert "林" in result.output

        # 検証
        list_result = runner.invoke(main, ["list"])
        assert "渡辺" in list_result.output
        assert "小川" in list_result.output
        assert "林" in list_result.output

        # 履歴も記録されている
        hist_result = runner.invoke(main, ["history"])
        assert "SPEAKER_A → 渡辺" in hist_result.output

    def test_correct_skips_missing_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "db.sqlite3"))
        from voice_shiwake import voiceprint as vp_mod
        monkeypatch.setattr(vp_mod, "compute_embedding", lambda p: np.array([1.0, 0.0]))

        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()
        (samples_dir / "SPEAKER_A.wav").touch()
        # SPEAKER_B.wav は存在しない

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "correct",
                "--samples-dir", str(samples_dir),
                "--map", "A=渡辺",
                "--map", "B=小川",
            ],
        )
        # A は登録、B は SKIP
        assert "[OK]" in result.output
        assert "[SKIP]" in result.output

    def test_correct_invalid_map_format_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_DB_PATH", str(tmp_path / "db.sqlite3"))
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["correct", "--samples-dir", str(samples_dir), "--map", "Aは田中"],
        )
        assert result.exit_code == 1

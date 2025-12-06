import builtins
import importlib
import logging

import pytest
from click.testing import CliRunner


def _block_pyannote_import(monkeypatch, exc: Exception) -> None:
    """Force any import of pyannote.* to raise the provided exception."""
    real_import = builtins.__import__

    def _patched(name, *args, **kwargs):
        if name.startswith("pyannote"):
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched)


def test_help_runs_without_pyannote(monkeypatch):
    from asr_cli import cli as cli_module

    _block_pyannote_import(monkeypatch, ImportError("pyannote unavailable"))
    importlib.reload(cli_module)

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "asr-mps" in result.output


def test_load_pyannote_failure_logs_warning_and_exits(monkeypatch, caplog):
    import asr_cli.diarization as d

    _block_pyannote_import(
        monkeypatch,
        AttributeError("module 'torchaudio' has no attribute 'AudioMetaData'"),
    )
    monkeypatch.setattr(d, "_PYANNOTE_IMPORT_ERROR", None)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(SystemExit):
            d.load_pyannote(None)

    assert any("pyannote.audio import failed" in rec.message for rec in caplog.records)

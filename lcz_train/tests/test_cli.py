"""CLI smoke tests: splits command runs end-to-end; run/mosaic wiring imports cleanly."""

import json

from typer.testing import CliRunner

from lcz_train.cli import app

runner = CliRunner()


def test_splits_command_writes_json(tmp_path):
    out = tmp_path / "splits.json"
    result = runner.invoke(app, [
        "splits",
        "--aoi", "Nairobi", "--aoi", "Toronto",
        "--so2sat-aoi", "Nairobi",
        "--version", "v1", "--seed", "0",
        "--bounds-csv", "data/guppd_bounds.csv",
        "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["test"] == ["Nairobi"]
    assert "Toronto" in data["train"] + data["val"]


def test_run_command_is_registered():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--exp" in result.output


def test_mosaic_command_is_registered():
    result = runner.invoke(app, ["mosaic", "--help"])
    assert result.exit_code == 0
    assert "--aoi" in result.output

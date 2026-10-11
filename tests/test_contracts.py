import json

from autoinsight.agents.contracts import normalise_stats


def test_normalise_stats_handles_read_only_stats_file(tmp_path):
    out_dir = tmp_path / "out"
    charts = out_dir / "charts"
    charts.mkdir(parents=True)
    (charts / "b.png").write_bytes(b"x")
    (charts / "a.png").write_bytes(b"x")

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps({"top_drivers": [{"feature": "x", "strength": "0.7", "finding": "f"}]}), encoding="utf-8")
    stats_path.chmod(0o444)

    normalise_stats(out_dir)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["charts"] == ["a.png", "b.png"]
    assert stats["top_drivers"][0]["strength"] == 0.7

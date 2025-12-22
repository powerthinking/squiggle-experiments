import os


def test_scout_writes_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUIGGLE_DATA_ROOT", str(tmp_path))

    cfg_path = tmp_path / "scout_test.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "run_id: golden_scout_smoke_s1337",
                "run_name: scout_smoke",
                "seed: 1337",
                "steps: 25",
                "batch_size: 32",
                "lr: 0.001",
                "device: cpu",
                "model:",
                "  d_model: 64",
                "  n_layers: 2",
                "  n_heads: 2",
                "  d_ff: 128",
                "  dropout: 0.0",
                "task:",
                "  p: 31",
                "capture:",
                "  every_steps: 10",
                "  layers: [0, 1]",
                "  embeddings: true",
                "  residuals: true",
                "  source: probe_fixed",
                "probes:",
                "  fixed:",
                "    enabled: true",
                "    n_examples: 32",
                "    seed: 123",
                "  holdout:",
                "    enabled: false",
                "probe_eval:",
                "  every_steps: 10",
                "triggers:",
                "  enabled: false",
                "",
            ]
        )
    )

    from squiggle_core import paths
    from squiggle_experiments.scout.run import run_scout

    run_id = run_scout(str(cfg_path))
    assert run_id == "golden_scout_smoke_s1337"

    assert (paths.run_dir(run_id) / "meta.json").exists()
    assert paths.captures_dir(run_id).exists()
    assert any(paths.captures_dir(run_id).glob("step_*"))
    assert paths.metrics_wide_path(run_id).exists()
    assert paths.metrics_scalar_path(run_id).exists()

    assert str(paths.run_dir(run_id)).startswith(str(tmp_path))

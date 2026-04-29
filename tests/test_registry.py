from pathlib import Path

from rare_synth.pipeline.registry import append_registry_row, load_registry


def test_registry_append_and_load(tmp_path: Path):
    root = tmp_path
    append_registry_row(
        root=root,
        run_id='run-1',
        cohort_name='test-cohort',
        config_path='configs/test.yaml',
        stage='train',
        status='success',
    )
    df = load_registry(root)
    assert len(df) == 1
    assert df.iloc[0]['run_id'] == 'run-1'
    assert df.iloc[0]['stage'] == 'train'

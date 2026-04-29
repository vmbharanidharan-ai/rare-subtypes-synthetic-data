from rare_synth.config import load_config


def test_load_default_uvm_config():
    cfg = load_config('configs/default_uvm.yaml')
    assert cfg.cohort_name == 'uveal-melanoma'
    assert cfg.data_source == 'gdc'
    assert cfg.train_model in {'ctgan', 'tvae'}
    assert isinstance(cfg.baseline_models, list)
    assert 'ctgan' in cfg.baseline_models

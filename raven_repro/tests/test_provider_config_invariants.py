import pytest

from raven.eval_protocol import require_uniform_provider_config


def test_uniform_tr_provider_is_canonicalized_once():
    rows = [{"w_seed": "7", "w_channel": "3"}, {"w_seed": 7, "w_channel": 3}]
    config, config_hash = require_uniform_provider_config("TR", rows)
    assert config["w_seed"] == 7
    assert len(config_hash) == 64


def test_tr_provider_seed_drift_fails_fast():
    with pytest.raises(ValueError, match="mixed provider configs"):
        require_uniform_provider_config("TR", [{"w_seed": 7}, {"w_seed": 8}])


def test_nested_formal_provider_config_is_honored():
    config, _ = require_uniform_provider_config(
        "TR", [{"provider_config": {"w_seed": 11}}, {"provider_config": '{"w_seed":11}'}]
    )
    assert config["w_seed"] == 11

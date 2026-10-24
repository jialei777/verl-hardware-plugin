# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the TPU engine options read from ``TorchtitanEngineConfig.engine_kwargs.tpu``."""

import logging
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import TorchtitanEngineConfig
from verl_hardware_plugin.accelerators.tpu.engines import engine_options
from verl_hardware_plugin.accelerators.tpu.engines.engine_options import TPUEngineOptions, get_tpu_engine_options
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    extend_torchtitan_engine_config,
    get_tpu_seq_bucket_size,
    resolve_tpu_torchtitan_options,
    resolve_tpu_use_fused_linear_cross_entropy,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env_name in engine_options._LEGACY_ENV_VARS.values():
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(engine_options, "_warned", set())


def _cfg(tpu=None, **fields):
    return SimpleNamespace(engine_kwargs={"tpu": tpu} if tpu is not None else {}, **fields)


def test_engine_kwargs_are_parsed_and_typed():
    options = get_tpu_engine_options(
        _cfg(
            {
                "use_splash_attention": "true",
                "use_simple_fsdp": False,
                "fuse_qkv": "false",
                "tpu_eager_mode": "DEFER_NEVER",
                "optim_impl": "for-loop",
                "seq_bucket_size": "512",
            }
        )
    )
    assert options == TPUEngineOptions(
        use_splash_attention=True,
        use_simple_fsdp=False,
        fuse_qkv=False,
        tpu_eager_mode="DEFER_NEVER",
        optim_impl="for-loop",
        seq_bucket_size=512,
    )
    assert get_tpu_engine_options(_cfg()) == TPUEngineOptions()
    assert get_tpu_engine_options(None) == TPUEngineOptions()


@pytest.mark.parametrize(
    ("tpu", "match"),
    [
        ({"use_splash_attn": True}, "Unknown TPU engine option"),
        ({"use_simple_fsdp": "maybe"}, "Invalid TPU engine option use_simple_fsdp"),
        ({"seq_bucket_size": 0}, "Invalid TPU engine option seq_bucket_size"),
    ],
)
def test_bad_engine_kwargs_raise(tpu, match):
    with pytest.raises(ValueError, match=match):
        get_tpu_engine_options(_cfg(tpu))


def test_resolvers_read_engine_kwargs():
    compiled = _cfg({"tpu_eager_mode": "NONE", "use_simple_fsdp": False}, use_torch_compile=True)
    assert resolve_tpu_torchtitan_options(compiled) == (True, False, None)
    assert resolve_tpu_use_fused_linear_cross_entropy(compiled) is True
    no_fused = _cfg({"use_fused_linear_cross_entropy": False}, use_torch_compile=True)
    assert resolve_tpu_use_fused_linear_cross_entropy(no_fused) is False
    assert get_tpu_seq_bucket_size(_cfg({"seq_bucket_size": 128}, pad_to_length=True, pad_to_length_bucket=512)) == 128
    assert get_tpu_seq_bucket_size(_cfg(pad_to_length=True, pad_to_length_bucket=512)) == 512


def test_legacy_field_and_env_still_work_with_a_warning(monkeypatch, caplog):
    legacy = _cfg(use_torch_compile=True, use_splash_attention=None, use_simple_fsdp=False, tpu_eager_mode=None)
    with caplog.at_level(logging.WARNING, logger=engine_options.__name__):
        assert get_tpu_engine_options(legacy).use_simple_fsdp is False
    assert "torchtitan.use_simple_fsdp is deprecated" in caplog.text

    # engine_kwargs wins over the deprecated field; the environment still wins over both, as before.
    both = _cfg({"use_simple_fsdp": True}, use_simple_fsdp=False)
    assert get_tpu_engine_options(both).use_simple_fsdp is True
    monkeypatch.setenv("VERL_TPU_USE_SIMPLE_FSDP", "0")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=engine_options.__name__):
        assert get_tpu_engine_options(both).use_simple_fsdp is False
    assert "VERL_TPU_USE_SIMPLE_FSDP='0'" in caplog.text and "deprecated" in caplog.text


def test_engine_kwargs_survive_verl_config_conversion():
    """Hydra/OmegaConf -> TorchtitanEngineConfig, as verl builds the engine config for every worker."""
    extend_torchtitan_engine_config()
    raw = OmegaConf.create(
        {
            "_target_": "verl.workers.config.TorchtitanEngineConfig",
            "use_torch_compile": True,
            "engine_kwargs": {"tpu": {"use_simple_fsdp": False, "tpu_eager_mode": "DEFER_NEVER"}},
        }
    )
    cfg = omega_conf_to_dataclass(raw)
    assert isinstance(cfg, TorchtitanEngineConfig)
    assert resolve_tpu_torchtitan_options(cfg) == (True, False, "DEFER_NEVER")
    assert omega_conf_to_dataclass(OmegaConf.create({"_target_": raw._target_})).engine_kwargs == {}

import importlib

import pytest
import torch

from refua.boltz.api import Boltz2


def _stub_boltz(*, use_kernels: bool = True) -> Boltz2:
    boltz = object.__new__(Boltz2)
    boltz.use_kernels = use_kernels
    boltz.model = None
    boltz._affinity_model = None
    return boltz


class _CueqMissingThenSuccess:
    def __init__(self) -> None:
        self.use_kernels = True
        self.calls = 0

    def __call__(self, _batch, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            exc = ModuleNotFoundError("No module named 'cuequivariance_torch'")
            exc.name = "cuequivariance_torch"
            raise exc
        return {"ok": torch.tensor([1.0])}


class _OtherMissingModule:
    def __init__(self) -> None:
        self.use_kernels = True

    def __call__(self, _batch, **_kwargs):
        importlib.import_module("refua_missing_module_for_kernel_fallback_test")


def test_run_model_retries_without_kernels_when_cueq_is_missing() -> None:
    boltz = _stub_boltz(use_kernels=True)
    model = _CueqMissingThenSuccess()
    boltz.model = model

    with pytest.warns(RuntimeWarning, match="use_kernels=False"):
        out = boltz._run_model(
            model,
            {},
            recycling_steps=0,
            sampling_steps=1,
            diffusion_samples=1,
            max_parallel_samples=1,
        )

    assert model.calls == 2
    assert boltz.use_kernels is False
    assert model.use_kernels is False
    assert out["ok"].item() == pytest.approx(1.0)


def test_run_model_does_not_mask_unrelated_missing_modules() -> None:
    boltz = _stub_boltz(use_kernels=True)
    model = _OtherMissingModule()
    boltz.model = model

    with pytest.raises(
        ModuleNotFoundError, match="refua_missing_module_for_kernel_fallback_test"
    ):
        boltz._run_model(
            model,
            {},
            recycling_steps=0,
            sampling_steps=1,
            diffusion_samples=1,
            max_parallel_samples=1,
        )

    assert boltz.use_kernels is True
    assert model.use_kernels is True


def test_run_model_uses_model_kernel_flag_for_fallback() -> None:
    boltz = _stub_boltz(use_kernels=False)
    model = _CueqMissingThenSuccess()
    boltz.model = model

    with pytest.warns(RuntimeWarning, match="use_kernels=False"):
        out = boltz._run_model(
            model,
            {},
            recycling_steps=0,
            sampling_steps=1,
            diffusion_samples=1,
            max_parallel_samples=1,
        )

    assert model.calls == 2
    assert boltz.use_kernels is False
    assert model.use_kernels is False
    assert out["ok"].item() == pytest.approx(1.0)

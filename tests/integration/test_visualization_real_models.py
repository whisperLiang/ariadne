from __future__ import annotations

import os

import pytest
import torch

from ariadne import SplitSpec, prepare_split
from ariadne.visualization import export_split_dot, export_trace_dot

pytestmark = pytest.mark.integration


def _real_models_enabled() -> bool:
    return os.environ.get("ARIADNE_RUN_REAL_MODELS") == "1"


@pytest.mark.skipif(not _real_models_enabled(), reason="set ARIADNE_RUN_REAL_MODELS=1")
def test_torchvision_resnet18_split_visualization_dot() -> None:
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    runtime = prepare_split(
        resnet18(weights=None).eval(),
        example_inputs=(torch.randn(2, 3, 64, 64),),
        split=SplitSpec(
            boundary="after:layer1",
            dynamic_batch=(2, 3),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    dot = export_split_dot(runtime.trace_plan, runtime.candidate)

    assert runtime.candidate.split_id in dot
    assert "layer1" in dot
    assert "Conv2d" in dot
    assert "BatchNorm2d" in dot
    assert "boundary_bytes" in dot
    assert 'ariadne_split_role="prefix"' in dot
    assert 'ariadne_split_role="suffix"' in dot
    assert 'ariadne_split_role="boundary"' in dot


@pytest.mark.skipif(not _real_models_enabled(), reason="set ARIADNE_RUN_REAL_MODELS=1")
def test_torchvision_vgg11_trace_and_render_visualization_dot() -> None:
    pytest.importorskip("torchvision")
    pytest.importorskip("graphviz")
    from torchvision.models import vgg11

    from ariadne.visualization import render_split_graph

    runtime = prepare_split(
        vgg11(weights=None).eval(),
        example_inputs=(torch.randn(2, 3, 64, 64),),
        split=SplitSpec(
            boundary="after:features.3",
            dynamic_batch=(2, 3),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    trace_dot = export_trace_dot(runtime.trace_plan)
    split_dot = render_split_graph(runtime.trace_plan, runtime.candidate, return_dot=True)

    assert "features.0" in trace_dot
    assert "features.3" in trace_dot
    assert trace_dot.count('label="features.2  #') == 1
    assert trace_dot.count("max_pool2d_with_indices.default") == 5
    assert trace_dot.count("Linear") == 3
    assert "t.default" not in trace_dot
    assert "detach.default" not in trace_dot
    assert "Conv2d" in trace_dot
    assert "ReLU" in trace_dot
    assert "call_function" not in trace_dot
    assert split_dot is not None
    assert runtime.candidate.split_id in split_dot
    assert "features.3" in split_dot
    assert "boundary_bytes" in split_dot

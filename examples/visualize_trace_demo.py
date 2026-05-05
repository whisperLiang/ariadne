from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ariadne import SplitSpec, prepare_split


@dataclass(frozen=True)
class VisualizationCase:
    name: str
    model_factory: Callable[[], nn.Module]
    input_shape: tuple[int, ...]
    boundary: str


def _resnet18_case() -> VisualizationCase:
    from torchvision.models import resnet18

    return VisualizationCase(
        name="resnet18",
        model_factory=lambda: resnet18(weights=None).eval(),
        input_shape=(3, 64, 64),
        boundary="after:layer1",
    )


def _vgg11_case() -> VisualizationCase:
    from torchvision.models import vgg11

    return VisualizationCase(
        name="vgg11",
        model_factory=lambda: vgg11(weights=None).eval(),
        input_shape=(3, 64, 64),
        boundary="after:features.3",
    )


def _cases(selection: str) -> tuple[VisualizationCase, ...]:
    available = {
        "resnet18": _resnet18_case,
        "vgg11": _vgg11_case,
    }
    if selection == "all":
        return tuple(factory() for factory in available.values())
    return (available[selection](),)


def visualize_case(case: VisualizationCase) -> None:
    runtime = prepare_split(
        case.model_factory(),
        example_inputs=(torch.randn(2, *case.input_shape),),
        split=SplitSpec(
            boundary=case.boundary,
            dynamic_batch=(2, 3),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    trace_path = f"{case.name}_trace_graph"
    split_path = f"{case.name}_split_graph"
    runtime.visualize(view="trace", outpath=trace_path, fileformat="svg")
    runtime.visualize(view="split", outpath=split_path, fileformat="svg")
    print(
        f"{case.name}: split_id={runtime.split_id} "
        f"nodes={len(runtime.trace_plan.nodes)} wrote {trace_path}.svg and {split_path}.svg"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Ariadne TracePlan and split graphs for real torchvision models."
    )
    parser.add_argument(
        "--model",
        choices=["resnet18", "vgg11", "all"],
        default="resnet18",
        help="Real model to visualize.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    for case in _cases(args.model):
        visualize_case(case)


if __name__ == "__main__":
    main()

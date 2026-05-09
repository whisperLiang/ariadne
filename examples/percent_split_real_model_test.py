from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ariadne import SplitSpec
from ariadne.api import _prepare_replay_runtime_from_plan, _prepare_runtime_from_plan
from ariadne.trace.tracer import trace_model
from examples.real_model_functional_test import (
    RFDETRTensorWrapper,
    assert_nested_close,
    assert_split_train_equivalent,
)

PERCENTAGES = tuple(range(10, 100, 10))


@dataclass(frozen=True)
class ModelCase:
    key: str
    name: str
    input_shape: tuple[int, ...]
    builder: Callable[[], nn.Module]


@dataclass(frozen=True)
class PercentSplitRow:
    model: str
    percent: int
    replay: str
    train: str
    split_id: str
    boundary_nodes: int
    trace_nodes: int
    detail: str = ""


def run_percent_split_matrix(
    *,
    model_keys: tuple[str, ...],
    percentages: tuple[int, ...],
    trace_batch: int,
    test_batch: int,
    device: torch.device,
    skip_replay: bool,
    skip_train: bool,
    train_rtol: float,
    train_atol: float,
) -> list[PercentSplitRow]:
    rows: list[PercentSplitRow] = []
    for case in _expand_model_cases(model_keys):
        rows.extend(
            _run_model_case(
                case,
                percentages=percentages,
                trace_batch=trace_batch,
                test_batch=test_batch,
                device=device,
                skip_replay=skip_replay,
                skip_train=skip_train,
                train_rtol=train_rtol,
                train_atol=train_atol,
            )
        )
    return rows


def _run_model_case(
    case: ModelCase,
    *,
    percentages: tuple[int, ...],
    trace_batch: int,
    test_batch: int,
    device: torch.device,
    skip_replay: bool,
    skip_train: bool,
    train_rtol: float,
    train_atol: float,
) -> list[PercentSplitRow]:
    print(f"\n== {case.name} ==")
    model = case.builder().eval().to(device)
    trace_inputs = _make_inputs(trace_batch, case.input_shape, device)
    replay_inputs = _make_inputs(test_batch, case.input_shape, device)
    plan = trace_model(
        model,
        example_inputs=(trace_inputs,),
        dynamic_batch=(trace_batch, test_batch),
        trace_batch_mode="batch_gt1",
    )

    rows: list[PercentSplitRow] = []
    for percent in percentages:
        split = SplitSpec(
            boundary=f"percent:{percent}",
            dynamic_batch=(trace_batch, test_batch),
            trainable=True,
            trace_batch_mode="batch_gt1",
        )
        replay_status = "SKIP"
        train_status = "SKIP"
        split_id = ""
        boundary_nodes = 0
        details: list[str] = []
        if not skip_replay:
            try:
                replay_runtime = _prepare_replay_runtime_from_plan(
                    plan,
                    spec=split,
                    split=split,
                    objective=None,
                    mode="generated_eager",
                    compile_options=None,
                    validation="strict",
                    materialize_boundary=True,
                )
                split_id = replay_runtime.split_id
                boundary_nodes = len(replay_runtime.boundary_order)
                with torch.no_grad():
                    actual = replay_runtime.run_suffix(replay_runtime.run_prefix(replay_inputs))
                    expected = model(replay_inputs)
                assert_nested_close(actual, expected)
                replay_status = "PASS"
            except Exception as error:  # noqa: BLE001
                replay_status = "FAIL"
                details.append(f"replay {error.__class__.__name__}: {error}")
                traceback.print_exc()

        if not skip_train:
            try:
                train_runtime = _prepare_runtime_from_plan(
                    plan,
                    spec=split,
                    split=split,
                    objective=None,
                    mode="generated_eager",
                    compile_options=None,
                )
                split_id = split_id or train_runtime.split_id
                boundary_nodes = boundary_nodes or len(train_runtime.segments.boundary_order)
                assert_split_train_equivalent(
                    model,
                    train_runtime,
                    _make_inputs(test_batch, case.input_shape, device),
                    rtol=train_rtol,
                    atol=train_atol,
                )
                train_status = "PASS"
            except Exception as error:  # noqa: BLE001
                train_status = "FAIL"
                details.append(f"train {error.__class__.__name__}: {error}")
                traceback.print_exc()

        row = PercentSplitRow(
            model=case.name,
            percent=percent,
            replay=replay_status,
            train=train_status,
            split_id=split_id,
            boundary_nodes=boundary_nodes,
            trace_nodes=len(plan.nodes),
            detail="; ".join(details),
        )
        rows.append(row)
        print(_format_row(row))
    return rows


def _make_inputs(
    batch_size: int,
    input_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(batch_size, *input_shape, device=device)


def _expand_model_cases(keys: tuple[str, ...]) -> tuple[ModelCase, ...]:
    registry = _model_registry()
    expanded: list[str] = []
    for key in keys:
        if key == "all":
            expanded.extend(["rfdetr", "tinynext", "yolov8n", "resnet18", "resnet50"])
        elif key == "complex":
            expanded.extend(
                [
                    "swin_tiny",
                    "vit_base",
                    "convnext_tiny",
                    "efficientnetv2_s",
                    "deeplabv3_resnet50",
                    "maxvit_tiny",
                ]
            )
        elif key == "resnet":
            expanded.extend(["resnet18", "resnet50"])
        elif key == "yolo":
            expanded.append("yolov8n")
        else:
            expanded.append(key)

    seen: set[str] = set()
    cases: list[ModelCase] = []
    for key in expanded:
        if key in seen:
            continue
        seen.add(key)
        cases.append(registry[key])
    return tuple(cases)


def _model_registry() -> dict[str, ModelCase]:
    return {
        "rfdetr": ModelCase(
            key="rfdetr",
            name="RF-DETR Nano",
            input_shape=(3, 128, 128),
            builder=lambda: RFDETRTensorWrapper().eval(),
        ),
        "tinynext": ModelCase(
            key="tinynext",
            name="TinyNeXt",
            input_shape=(3, 128, 128),
            builder=_build_tinynext,
        ),
        "yolov8n": ModelCase(
            key="yolov8n",
            name="YOLOv8n",
            input_shape=(3, 64, 64),
            builder=_build_yolov8n,
        ),
        "resnet18": ModelCase(
            key="resnet18",
            name="torchvision resnet18",
            input_shape=(3, 96, 96),
            builder=_build_resnet18,
        ),
        "resnet50": ModelCase(
            key="resnet50",
            name="timm resnet50",
            input_shape=(3, 96, 96),
            builder=_build_resnet50,
        ),
        "swin_tiny": ModelCase(
            key="swin_tiny",
            name="timm swin_tiny_patch4_window7_224",
            input_shape=(3, 224, 224),
            builder=lambda: _build_timm_model("swin_tiny_patch4_window7_224"),
        ),
        "vit_base": ModelCase(
            key="vit_base",
            name="timm vit_base_patch16_224",
            input_shape=(3, 224, 224),
            builder=lambda: _build_timm_model("vit_base_patch16_224"),
        ),
        "convnext_tiny": ModelCase(
            key="convnext_tiny",
            name="timm convnext_tiny",
            input_shape=(3, 128, 128),
            builder=lambda: _build_timm_model("convnext_tiny"),
        ),
        "efficientnetv2_s": ModelCase(
            key="efficientnetv2_s",
            name="timm efficientnetv2_s",
            input_shape=(3, 128, 128),
            builder=lambda: _build_timm_model("efficientnetv2_s"),
        ),
        "deeplabv3_resnet50": ModelCase(
            key="deeplabv3_resnet50",
            name="torchvision deeplabv3_resnet50",
            input_shape=(3, 96, 96),
            builder=_build_deeplabv3_resnet50,
        ),
        "maxvit_tiny": ModelCase(
            key="maxvit_tiny",
            name="timm maxvit_tiny_tf_224",
            input_shape=(3, 224, 224),
            builder=lambda: _build_timm_model("maxvit_tiny_tf_224"),
        ),
    }


def _build_tinynext() -> nn.Module:
    import timm

    errors: list[str] = []
    for model_name in ("tinynext", "inception_next_tiny", "convnext_tiny"):
        try:
            model = timm.create_model(model_name, pretrained=False)
            print(f"tinynext builder resolved to timm model {model_name!r}")
            return model.eval()
        except Exception as error:  # noqa: BLE001
            errors.append(f"{model_name}: {error}")
    raise RuntimeError("Could not build a TinyNeXt-compatible timm model: " + "; ".join(errors))


def _build_yolov8n() -> nn.Module:
    from ultralytics import YOLO

    weights_dir = Path(".ariadne_models")
    weights_dir.mkdir(exist_ok=True)
    with _pushd(weights_dir):
        yolo = YOLO("yolov8n.pt")
    return yolo.model.eval()


def _build_resnet18() -> nn.Module:
    from torchvision.models import resnet18

    return resnet18(weights=None).eval()


def _build_resnet50() -> nn.Module:
    return _build_timm_model("resnet50")


def _build_timm_model(model_name: str) -> nn.Module:
    import timm

    return timm.create_model(model_name, pretrained=False).eval()


def _build_deeplabv3_resnet50() -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    return deeplabv3_resnet50(weights=None, weights_backbone=None).eval()


@contextlib.contextmanager
def _pushd(path: Path) -> Any:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _format_row(row: PercentSplitRow) -> str:
    detail = f" detail={row.detail}" if row.detail else ""
    return (
        f"{row.percent:>3}% replay={row.replay:<4} train={row.train:<4} "
        f"split={row.split_id or '-'} boundary_nodes={row.boundary_nodes}{detail}"
    )


def _print_table(rows: Iterable[PercentSplitRow]) -> None:
    print(
        "\n| model | percent | replay | train | split_id | "
        "boundary_nodes | trace_nodes | detail |"
    )
    print("|---|---:|---|---|---|---:|---:|---|")
    for row in rows:
        print(
            f"| {row.model} | {row.percent} | {row.replay} | {row.train} | "
            f"{row.split_id or '-'} | {row.boundary_nodes} | {row.trace_nodes} | "
            f"{row.detail} |"
        )


def _parse_percentages(values: list[str] | None) -> tuple[int, ...]:
    if not values:
        return PERCENTAGES
    percentages: list[int] = []
    for value in values:
        for item in value.split(","):
            percent = int(item.strip().removesuffix("%"))
            if not 0 < percent < 100:
                raise argparse.ArgumentTypeError("percentages must be between 1 and 99")
            percentages.append(percent)
    return tuple(percentages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "complex",
            "rfdetr",
            "tinynext",
            "yolo",
            "yolov8n",
            "resnet",
            "resnet18",
            "resnet50",
            "swin_tiny",
            "vit_base",
            "convnext_tiny",
            "efficientnetv2_s",
            "deeplabv3_resnet50",
            "maxvit_tiny",
        ],
    )
    parser.add_argument("--percentages", nargs="*", default=None)
    parser.add_argument("--trace-batch", type=int, default=2)
    parser.add_argument("--test-batch", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-rtol", type=float, default=1e-3)
    parser.add_argument("--train-atol", type=float, default=1e-4)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    rows = run_percent_split_matrix(
        model_keys=tuple(args.models),
        percentages=_parse_percentages(args.percentages),
        trace_batch=args.trace_batch,
        test_batch=args.test_batch,
        device=device,
        skip_replay=args.skip_replay,
        skip_train=args.skip_train,
        train_rtol=args.train_rtol,
        train_atol=args.train_atol,
    )
    _print_table(rows)

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps([asdict(row) for row in rows], indent=2),
            encoding="utf-8",
        )

    failures = [row for row in rows if row.replay == "FAIL" or row.train == "FAIL"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

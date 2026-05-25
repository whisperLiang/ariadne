from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ariadne.pattern.split_spec import SplitSpec  # noqa: E402
from ariadne.planner.candidate_validation import validate_split_candidates  # noqa: E402
from ariadne.planner.frontier import SplitCandidate, enumerate_frontier_splits  # noqa: E402
from ariadne.trace.tracer import trace_model  # noqa: E402


@dataclass(frozen=True)
class ModelCase:
    name: str
    model: nn.Module
    input_shape: tuple[int, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every Ariadne frontier candidate for real models with "
            "split replay and synthetic split training."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(_MODEL_FACTORIES),
        help="Model case to run. Can be repeated. Defaults to all lightweight cases.",
    )
    parser.add_argument("--low", type=int, default=1)
    parser.add_argument("--trace-batch", type=int, default=2)
    parser.add_argument("--high", type=int, default=3)
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Include YOLO, Swin Tiny, DeepLabV3, and RF-DETR when --model is omitted.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional development cap; omitted means validate all candidates.",
    )
    args = parser.parse_args()

    names = args.model
    if names is None:
        names = list(_LIGHTWEIGHT_DEFAULTS)
        if args.include_heavy:
            names = list(_MODEL_FACTORIES)

    for name in names:
        case = _MODEL_FACTORIES[name]()
        _validate_case(
            case,
            low=args.low,
            trace_batch=args.trace_batch,
            high=args.high,
            max_candidates=args.max_candidates,
        )


def _validate_case(
    case: ModelCase,
    *,
    low: int,
    trace_batch: int,
    high: int,
    max_candidates: int | None,
) -> None:
    torch.manual_seed(0)
    model = case.model.eval()
    example_inputs = (torch.randn(trace_batch, *case.input_shape),)
    spec = SplitSpec(
        boundary="auto",
        dynamic_batch=(low, high),
        trainable=True,
        trace_batch_mode="batch_gt1",
    )
    print(f"\n== {case.name} ==")
    start = time.perf_counter()
    plan = trace_model(
        model,
        example_inputs=example_inputs,
        dynamic_batch=spec.dynamic_batch,
        trace_batch_mode=spec.trace_batch_mode,
    )
    candidates = enumerate_frontier_splits(plan)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    print(
        f"trace_nodes={len(plan.nodes)} candidates={len(candidates)} "
        f"dynamic_batch={spec.dynamic_batch}"
    )

    accepted: list[SplitCandidate] = []
    rejected: list[SplitCandidate] = []
    for index, candidate in enumerate(candidates, start=1):
        validated = validate_split_candidates(
            plan,
            spec=spec,
            example_inputs=example_inputs,
            candidates=(candidate,),
            require_training=True,
        )[0]
        status = "PASS" if validated.rejection_reason is None else "REJECT"
        print(f"[{index:04d}/{len(candidates):04d}] {status} {validated.split_id}")
        if validated.rejection_reason is None:
            accepted.append(validated)
        else:
            rejected.append(validated)

    elapsed = time.perf_counter() - start
    print(
        f"SUMMARY {case.name}: accepted={len(accepted)} rejected={len(rejected)} "
        f"elapsed={elapsed:.2f}s"
    )
    rejection_counts = Counter(_reason_bucket(item.rejection_reason) for item in rejected)
    for reason, count in rejection_counts.most_common():
        print(f"  rejected[{count}]: {reason}")


def _reason_bucket(reason: str | None) -> str:
    if reason is None:
        return "accepted"
    if ":" in reason:
        return reason.split(":", 1)[0]
    return reason


def _resnet18() -> ModelCase:
    from torchvision.models import resnet18

    return ModelCase("torchvision resnet18", resnet18(weights=None), (3, 64, 64))


def _vgg11() -> ModelCase:
    from torchvision.models import vgg11

    return ModelCase("torchvision vgg11", vgg11(weights=None), (3, 64, 64))


def _mobilenet_v3_large() -> ModelCase:
    from torchvision.models import mobilenet_v3_large

    return ModelCase(
        "torchvision mobilenet_v3_large",
        mobilenet_v3_large(weights=None),
        (3, 96, 96),
    )


def _timm_resnet50() -> ModelCase:
    import timm

    return ModelCase("timm resnet50", timm.create_model("resnet50", pretrained=False), (3, 96, 96))


def _timm_swin_tiny() -> ModelCase:
    import timm

    return ModelCase(
        "timm swin_tiny",
        timm.create_model("swin_tiny_patch4_window7_224", pretrained=False),
        (3, 224, 224),
    )


def _deeplabv3_resnet50() -> ModelCase:
    from torchvision.models.segmentation import deeplabv3_resnet50

    return ModelCase(
        "torchvision deeplabv3_resnet50",
        deeplabv3_resnet50(weights=None, weights_backbone=None),
        (3, 96, 96),
    )


def _yolo_v8n() -> ModelCase:
    from pathlib import Path

    from ultralytics import YOLO

    weights_dir = Path(".ariadne_models")
    weights_dir.mkdir(exist_ok=True)
    with _pushd(weights_dir):
        yolo = YOLO("yolov8n.pt")
    return ModelCase("YOLOv8n", yolo.model, (3, 64, 64))


def _rfdetr_nano() -> ModelCase:
    from examples.real_model_functional_test import RFDETRTensorWrapper

    return ModelCase("RF-DETR Nano", RFDETRTensorWrapper(), (3, 128, 128))


@contextlib.contextmanager
def _pushd(path: Any) -> Any:
    import os
    from pathlib import Path

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


_LIGHTWEIGHT_DEFAULTS = (
    "torchvision-resnet18",
    "torchvision-vgg11",
    "torchvision-mobilenet-v3-large",
    "timm-resnet50",
)

_MODEL_FACTORIES = {
    "torchvision-resnet18": _resnet18,
    "torchvision-vgg11": _vgg11,
    "torchvision-mobilenet-v3-large": _mobilenet_v3_large,
    "timm-resnet50": _timm_resnet50,
    "timm-swin-tiny": _timm_swin_tiny,
    "torchvision-deeplabv3-resnet50": _deeplabv3_resnet50,
    "yolo-v8n": _yolo_v8n,
    "rfdetr-nano": _rfdetr_nano,
}


if __name__ == "__main__":
    main()

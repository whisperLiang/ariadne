"""GPU Batch Performance Test - Full Version

Test model performance across different batch sizes (4, 32, 256) on GPU.
Trace phase uses batch_size=2, then tests split replay and training across batches.
Uses automatic split point selection to minimize boundary transfer overhead.

Supported models:
- ResNet50 (timm)
- MobileNetV3 Large (torchvision)
- EfficientNet-B0 (timm)
- YOLOv8n (ultralytics)
- Swin Transformer Tiny (timm)
- DeepLabV3 ResNet50 (torchvision)
- RF-DETR Nano (rfdetr)
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ariadne import SplitSpec, prepare_split


def sync_cuda():
    """Synchronize CUDA operations"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(fn, iterations=10, warmup=3):
    """Benchmark function execution time"""
    # Warmup
    for _ in range(warmup):
        fn()
    sync_cuda()
    
    # Measure
    times = []
    for _ in range(iterations):
        sync_cuda()
        start = perf_counter()
        fn()
        sync_cuda()
        times.append((perf_counter() - start) * 1000.0)
    
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def nested_tensor_loss(value):
    """Compute loss for nested tensor structures"""
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        return value.float().square().mean()
    if isinstance(value, torch.Tensor):
        return torch.tensor(0.0, device=value.device)
    if isinstance(value, (tuple, list)):
        losses = [nested_tensor_loss(item) for item in value]
        return sum(loss for loss in losses if loss.numel() > 0)
    if isinstance(value, dict):
        losses = [nested_tensor_loss(item) for item in value.values()]
        return sum(loss for loss in losses if loss.numel() > 0)
    return torch.tensor(0.0)


@contextlib.contextmanager
def _pushd(path: Path):
    """Temporarily change working directory"""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_model(name, model, input_shape, trace_batch=2, test_batches=None, iterations=10, warmup=3):
    """Test a single model with automatic split selection"""
    if test_batches is None:
        test_batches = [4, 32, 256]

    print("=" * 80)
    print(f"Testing Model: {name}")
    print("=" * 80)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Trace with automatic split selection
    print(f"Trace Phase (batch_size={trace_batch}, auto split selection)...")
    trace_input = torch.randn(trace_batch, *input_shape, device=device)
    
    trace_start = perf_counter()
    runtime = prepare_split(
        model,
        example_inputs=(trace_input,),
        split=SplitSpec(
            boundary="auto",
            dynamic_batch=(min(test_batches), max(test_batches)),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
        objective={
            "minimize": "boundary_bytes",
            "constraints": {"trainable_suffix": True},
        },
    )
    sync_cuda()
    trace_time = (perf_counter() - trace_start) * 1000.0
    
    print(f"✓ Trace Complete: {trace_time:.2f} ms")
    print(f"  Split ID: {runtime.split_id}")
    print(f"  Boundary: {runtime.candidate.boundary_after}")
    print(f"  Boundary Bytes: {runtime.candidate.cost.boundary_bytes:,}")
    print(f"  Trace Nodes: {len(runtime.trace_plan.nodes)}")
    print(f"  Prefix Nodes: {runtime.candidate.cost.prefix_node_count}")
    print(f"  Suffix Nodes: {runtime.candidate.cost.suffix_node_count}")
    print()
    
    # Test across different batch sizes
    results = {}
    for batch_size in test_batches:
        print(f"Testing batch_size={batch_size}")
        print("-" * 80)
        
        test_input = torch.randn(batch_size, *input_shape, device=device)
        
        # Inference
        with torch.no_grad():
            avg, std = benchmark(
                lambda test_input=test_input: model(test_input),
                iterations,
                warmup,
            )
            print(f"  direct_forward:      {avg:>8.3f} ± {std:>6.3f} ms")
            results[batch_size] = {"direct": avg}
            
            avg, std = benchmark(
                lambda test_input=test_input: runtime.run_suffix(runtime.run_prefix(test_input)),
                iterations,
                warmup,
            )
            print(f"  prefix+suffix:       {avg:>8.3f} ± {std:>6.3f} ms")
            results[batch_size]["split"] = avg
        
        # Training
        def direct_train(test_input=test_input):
            model.zero_grad(set_to_none=True)
            inputs = test_input.detach().clone().requires_grad_(True)
            loss = nested_tensor_loss(model(inputs))
            loss.backward()
        
        def split_train(test_input=test_input):
            runtime.trace_plan.root_module.zero_grad(set_to_none=True)
            inputs = test_input.detach().clone().requires_grad_(True)
            boundary = runtime.run_training_prefix(inputs)
            loss, boundary_grads = runtime.train_suffix(
                boundary,
                None,
                loss_fn=lambda outputs, _: nested_tensor_loss(outputs),
            )
            runtime.backward_prefix(boundary, boundary_grads=boundary_grads)
        
        avg, std = benchmark(direct_train, iterations, warmup)
        print(f"  direct_train:        {avg:>8.3f} ± {std:>6.3f} ms")
        results[batch_size]["direct_train"] = avg
        
        avg, std = benchmark(split_train, iterations, warmup)
        print(f"  split_train:         {avg:>8.3f} ± {std:>6.3f} ms")
        results[batch_size]["split_train"] = avg
        
        print()
    
    # Performance Analysis
    print("Performance Analysis:")
    print("-" * 80)
    print("Inference Overhead (prefix+suffix vs direct_forward):")
    for batch_size in test_batches:
        overhead = (results[batch_size]["split"] / results[batch_size]["direct"] - 1) * 100
        print(f"  Batch {batch_size:>3}: {overhead:>6.2f}%")
    
    print("\nTraining Overhead (split_train vs direct_train):")
    for batch_size in test_batches:
        overhead = (
            results[batch_size]["split_train"] / results[batch_size]["direct_train"] - 1
        ) * 100
        print(f"  Batch {batch_size:>3}: {overhead:>6.2f}%")
    print()
    
    return results


def run_resnet50(trace_batch, test_batches, iterations, warmup):
    """Test ResNet50 with automatic split selection"""
    import timm
    model = timm.create_model("resnet50", pretrained=False)
    return run_model("ResNet50", model, (3, 96, 96), trace_batch, test_batches, iterations, warmup)


def run_mobilenet(trace_batch, test_batches, iterations, warmup):
    """Test MobileNetV3 with automatic split selection"""
    from torchvision.models import mobilenet_v3_large
    model = mobilenet_v3_large(weights=None)
    return run_model(
        "MobileNetV3",
        model,
        (3, 96, 96),
        trace_batch,
        test_batches,
        iterations,
        warmup,
    )


def run_efficientnet(trace_batch, test_batches, iterations, warmup):
    """Test EfficientNet with automatic split selection"""
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=False)
    return run_model(
        "EfficientNet-B0",
        model,
        (3, 96, 96),
        trace_batch,
        test_batches,
        iterations,
        warmup,
    )


def run_yolo(trace_batch, test_batches, iterations, warmup):
    """Test YOLOv8n with automatic split selection"""
    from ultralytics import YOLO
    weights_dir = Path(".ariadne_models")
    weights_dir.mkdir(exist_ok=True)
    with _pushd(weights_dir):
        yolo = YOLO("yolov8n.pt")
    model = yolo.model
    return run_model("YOLOv8n", model, (3, 64, 64), trace_batch, test_batches, iterations, warmup)


def run_swin(trace_batch, test_batches, iterations, warmup):
    """Test Swin Transformer with automatic split selection"""
    import timm
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    return run_model(
        "Swin-Tiny",
        model,
        (3, 224, 224),
        trace_batch,
        test_batches,
        iterations,
        warmup,
    )


def run_deeplabv3(trace_batch, test_batches, iterations, warmup):
    """Test DeepLabV3 with automatic split selection"""
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    return run_model("DeepLabV3", model, (3, 96, 96), trace_batch, test_batches, iterations, warmup)


def run_rfdetr(trace_batch, test_batches, iterations, warmup):
    """Test RF-DETR with automatic split selection"""
    from rfdetr import RFDETRNano
    from rfdetr.utilities.tensors import NestedTensor
    
    class RFDETRWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = RFDETRNano(pretrain_weights=None).model.model
        
        def forward(self, x):
            mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), 
                             dtype=torch.bool, device=x.device)
            return self.model(NestedTensor(x, mask))
    
    model = RFDETRWrapper()
    return run_model("RF-DETR", model, (3, 128, 128), trace_batch, test_batches, iterations, warmup)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "GPU Batch Performance Test - trace with batch=2, test different batch sizes "
            "with auto split selection"
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "resnet50",
            "mobilenet",
            "efficientnet",
            "yolo",
            "swin",
            "deeplabv3",
            "rfdetr",
            "all",
        ],
        default=["all"],
        help="Models to test (default: all)",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[4, 32, 256],
        help="Batch sizes to test (default: 4 32 256)",
    )
    parser.add_argument(
        "--trace-batch",
        type=int,
        default=2,
        help="Batch size for trace phase (default: 2)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per test (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    
    args = parser.parse_args()
    
    # Device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("GPU Batch Performance Test (Automatic Split Selection)")
    print("=" * 80)
    print()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Trace Batch: {args.trace_batch}")
    print(f"Test Batches: {args.batches}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print("Split Selection: Automatic (minimize boundary_bytes, trainable_suffix required)")
    print()
    
    torch.manual_seed(0)
    
    # Model mapping
    model_map = {
        "resnet50": run_resnet50,
        "mobilenet": run_mobilenet,
        "efficientnet": run_efficientnet,
        "yolo": run_yolo,
        "swin": run_swin,
        "deeplabv3": run_deeplabv3,
        "rfdetr": run_rfdetr,
    }
    
    # Select models
    models_to_test = list(model_map.keys()) if "all" in args.models else args.models
    
    # Run tests
    all_results = {}
    for model_name in models_to_test:
        try:
            print(f"\n{'=' * 80}")
            print(f"Starting Test: {model_name}")
            print(f"{'=' * 80}\n")
            
            results = model_map[model_name](
                args.trace_batch, args.batches, args.iterations, args.warmup
            )
            all_results[model_name] = results
            
        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Summary
    if all_results:
        print("\n" + "=" * 80)
        print("Overall Performance Comparison")
        print("=" * 80)
        print()
        
        for batch_size in args.batches:
            print(f"Batch {batch_size}:")
            print("-" * 80)
            print(f"{'Model':<20} {'Inference OH':<15} {'Training OH':<15}")
            print("-" * 80)
            
            for model_name, results in all_results.items():
                if batch_size in results:
                    inf_overhead = (
                        results[batch_size]["split"] / results[batch_size]["direct"] - 1
                    ) * 100
                    train_overhead = (
                        results[batch_size]["split_train"]
                        / results[batch_size]["direct_train"]
                        - 1
                    ) * 100
                    print(
                        f"{model_name:<20} {inf_overhead:>6.2f}%        "
                        f"{train_overhead:>6.2f}%"
                    )
            print()
    
    print("=" * 80)
    print("All Tests Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

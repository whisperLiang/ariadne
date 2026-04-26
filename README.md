# Ariadne

Ariadne is a dynamic-batch split replay runtime for PyTorch models. It traces
real PyTorch forward execution into a symbolic split plan, generates efficient
prefix/suffix segment callables, and supports boundary-based replay and split
training.

Ariadne uses TorchLens-style tracing for broad PyTorch model compatibility, but
it does not use node-by-node interpreted replay as the default runtime. Instead,
it lowers traced graphs into generated prefix/suffix segment callables.

## Why Ariadne Exists

Split inference and split training need a clean boundary between offline model
preparation and lightweight runtime execution. Ariadne prepares a symbolic
TracePlan, validates a declarative SplitSpec, chooses a valid frontier, and
generates executable segments so runtime work stays simple:

1. run the prefix segment
2. package a boundary payload
3. run the suffix segment
4. optionally train the suffix and backpropagate boundary gradients into the prefix

## Design Principles

- Trace observed PyTorch forward behavior while keeping default metadata light.
- Treat the first input tensor dimension as symbolic batch dimension `B`.
- Keep concrete batch size out of trace and runtime cache keys.
- Make split declarations explicit and verifiable.
- Use generated eager segments as the default execution path.
- Keep node-by-node interpretation only for `debug_interpreter` validation.
- Use `torch.compile` only as an optional optimizer for generated segments.

## Installation With uv

```bash
uv sync
```

Dependencies are declared only in `pyproject.toml`, and `uv.lock` is committed
for reproducible installs.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

Useful demo commands:

```bash
uv run python examples/split_inference_demo.py
uv run python examples/split_training_demo.py
```

Optional real-model smoke checks install the `integration` extra and may download
YOLO weights. The current integration suite covers `YOLOv8n`, `RF-DETR Nano`,
`timm resnet50`, `timm swin_tiny_patch4_window7_224`, `torchvision mobilenet_v3_large`,
and `torchvision deeplabv3_resnet50`:

```bash
uv run --extra integration python examples/real_model_functional_test.py
uv run --extra integration python examples/real_model_timing.py --iterations 3 --warmup 1
ARIADNE_RUN_REAL_MODELS=1 uv run --extra integration pytest tests/integration -m integration
```

## Basic Split Inference

```python
import torch
from ariadne import SplitSpec, prepare_split

spec = SplitSpec(
    boundary="after:layer3",
    batch_symbol="B",
    dynamic_batch=(2, 64),
    trainable=True,
    trace_batch_mode="batch_gt1",
)

runtime = prepare_split(
    model,
    example_inputs=(torch.randn(4, 5),),
    split=spec,
    mode="generated_eager",
)

boundary = runtime.run_prefix(torch.randn(8, 5))
output = runtime.run_suffix(boundary)
```

## Basic Split Training

```python
import torch.nn.functional as F

boundary = runtime.run_prefix(x_batch)
loss, boundary_grads = runtime.train_suffix(
    boundary,
    targets,
    loss_fn=F.mse_loss,
    optimizer=suffix_optimizer,
)
runtime.backward_prefix(
    x_batch,
    boundary_grads=boundary_grads,
    optimizer=prefix_optimizer,
)
```

## Dynamic Batch

By default, Ariadne treats the first dimension of the first tensor input as the
symbolic batch dimension `B`. If an example trace uses batch size 4, tensors like
`(4, 256, 14, 14)` are recorded as `("B", 256, 14, 14)`.

`SplitSpec.trace_batch_mode` makes the preparation strategy explicit:

- `batch_1`: requires `example_inputs` batch size 1 and a `dynamic_batch` range
  that includes 1. Ariadne performs prepare-time provenance probing and can
  prepare a batch>1 structural variant when singleton and non-singleton aten
  paths differ.
- `batch_gt1`: requires `example_inputs` batch size greater than 1 and a
  `dynamic_batch` range that starts at 2 or greater. Ariadne stays in the
  non-singleton regime, derives affine batch shapes such as `4*B`, and avoids
  the extra singleton structural variant used by `batch_1`.

For real YOLO and RF-DETR smoke tests, Ariadne uses `batch_gt1` mode and verifies
cross-batch split replay plus split training on batch sizes 2 and 3.

At runtime, Ariadne materializes `B` from the actual input batch size, validates
that it is inside `SplitSpec.dynamic_batch`, and checks that non-batch dimensions
match the prepared boundary schema. The concrete batch size is intentionally not
part of `RuntimeCacheKey`.

## Execution Modes

- `debug_interpreter`: slow interpreter execution for validation and debugging.
- `generated_eager`: default generated prefix/suffix segment execution.
- `compiled`: applies `torch.compile` to generated segments after preparation.

Example:

```python
runtime = prepare_split(
    model,
    example_inputs=(x,),
    split=spec,
    mode="compiled",
    compile_options={"backend": "inductor", "mode": "reduce-overhead", "dynamic": True},
)
```

## Benchmarking

`ariadne.benchmark` includes utilities for measuring direct PyTorch forward,
debug/generated/compiled prefix+suffix execution, suffix-only replay, suffix
training, and prefix backward. Results include average latency, total latency,
batch size, split id, execution mode, and optional CUDA peak memory.

## Current Limitations

- The default tracer uses `TorchDispatchMode` runtime interception and records the
  observed forward path.
- Alias, mutation, RNG, FLOP, and memory metadata are intentionally lightweight.
- Segment generation currently supports tensor boundaries from prepared observed
  paths, including a batch>1 structural variant when batch=1 tracing needs it.
- Dynamic non-batch dimensions are reserved for future SplitSpec extensions.
- Shape expressions cover direct and affine batch-derived dimensions; more
  complex non-affine shape arithmetic is still limited.

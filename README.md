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

## Installation

### Quick Start (Users)

Install Ariadne from PyPI with `uv`:

```bash
uv add ariadne-split
```

If you are not using a `uv` project, `pip install ariadne-split` works too.

After installation, you can import and use Ariadne:

```python
from ariadne import SplitSpec, prepare_split
import torch

# See usage examples in "Basic Split Inference" and "Basic Split Training" sections below
```

### Optional Dependencies

For real-model integration tests (YOLOv8n, RF-DETR, timm models, torchvision), install with the `integration` extra:

```bash
uv add "ariadne-split[integration]"
```

For operation-level TracePlan and split-candidate visualization, install with the
`visualization` extra:

```bash
uv add "ariadne-split[visualization]"
```

The visualization extra installs the Python `graphviz` package. Rendering SVG,
PDF, or PNG files also requires the Graphviz system executable on `PATH`. DOT
export does not require the system executable.

### Development Setup (Contributors)

If you're developing Ariadne or want to run the full test suite:

```bash
uv sync
```

Dependencies are declared only in `pyproject.toml`, and `uv.lock` is committed
for reproducible installs.

## Development & Testing

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
uv run --extra integration --extra visualization python examples/visualize_trace_demo.py --model resnet18
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

Visualization also has optional real-model smoke tests using torchvision
`resnet18` and `vgg11`:

```bash
ARIADNE_RUN_REAL_MODELS=1 uv run --extra integration --extra visualization pytest tests/integration/test_visualization_real_models.py -m integration
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

boundary = runtime.run_training_prefix(x_batch)
loss, boundary_grads = runtime.train_suffix(
    boundary,
    targets,
    loss_fn=F.mse_loss,
    optimizer=suffix_optimizer,
)
runtime.backward_prefix(
    boundary,
    boundary_grads=boundary_grads,
    optimizer=prefix_optimizer,
)
```

For split training, use `run_training_prefix()` so the boundary keeps the
original prefix autograd graph. Then `backward_prefix(boundary, ...)` applies
the suffix boundary gradients directly to that graph without recomputing prefix
operations. This matters for RNG-sensitive training operations such as
`nn.Dropout`, where recomputing prefix would sample a different random mask.
The lightweight `run_prefix()` path is still available for split replay and
inference-style boundary generation.

## Visualization

Ariadne can export offline operation-level views of a captured `TracePlan` and
the selected split candidate. Visualization reads metadata already stored in
`TracePlan`, `TraceNode`, and `SplitCandidate`; it does not re-trace the model,
does not run generated segments, and does not store real activations or tensors.

The most direct path is through a prepared runtime:

```python
runtime.visualize(view="trace", outpath="trace_graph", fileformat="svg")
runtime.visualize(view="split", outpath="split_graph", fileformat="svg")
```

For tests, notebooks, and debugging environments where the Graphviz system
binary is unavailable, request DOT source instead:

```python
trace_dot = runtime.visualize(view="trace", return_dot=True)
split_dot = runtime.visualize(view="split", return_dot=True)
```

The public export helpers can also be used directly:

```python
from ariadne.visualization import (
    export_split_candidates_table,
    export_split_dot,
    export_trace_dot,
)

trace_dot = export_trace_dot(runtime.trace_plan)
split_dot = export_split_dot(runtime.trace_plan, runtime.candidate)
candidates = export_split_candidates_table(runtime.trace_plan)
```

Visualizations default to a module-structure view inspired by TorchLens: traced
operations are folded back into `nn.Module` paths, and deep module hierarchies
are collapsed to a readable nesting depth by default. For example, ResNet-style
blocks render as `layer1.0` `BasicBlock` clusters containing `conv`, `bn`,
`relu`, residual `add`, and `downsample` nodes, instead of a raw ATen operator
stream or a single opaque block. Labels omit ATen targets, trace node indices,
mutation/debug markers, dtype, raw byte counts, and buffers by default. Node
rows stay compact: module/type on the first line, symbolic shape plus activation
memory in MB on the second line, and shortened parameter rows such as
`params: weight(64x3x7x7), bias(x64)` or `params: 74.0K` for explicitly
collapsed modules. Trainable parameters use parentheses and frozen parameters
use square brackets.

When you need low-level debugging, request the operation view explicitly:

```python
runtime.visualize(
    view="trace",
    view_detail="operation",
    show_operation_targets=True,
    show_debug_markers=True,
    show_node_indices=True,
    outpath="trace_ops",
)
```

Use `max_module_depth` to collapse deep model hierarchies into coarser module
nodes, similar to TorchLens nesting-depth controls. Pass `None` to expand the
nested module clusters:

```python
runtime.visualize(view="split", max_module_depth=2)  # collapse BasicBlock nodes
runtime.visualize(view="trace", max_module_depth=None)
```

Split visualizations mark prefix, suffix, boundary, and passthrough nodes and
include lightweight cost information such as `boundary_bytes`, prefix/suffix
node counts, and whether the suffix is trainable.

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

Ariadne lets users choose between non-compiled generated execution and
`torch.compile` optimized execution at preparation time:

- `debug_interpreter`: slow interpreter execution for validation and debugging.
- `generated_eager`: default generated prefix/suffix segment execution. Use this
  for short-lived scripts, CPU-only environments, correctness checks, or when
  the expected number of calls is too small to pay back compile cost.
- `compiled`: applies `torch.compile` to generated segments after preparation.
  Use this for long-lived GPU services or repeated same-model, same-shape
  workloads where startup warmup can happen outside the online request path.

Split retain training uses `prepare_split(...)` in either mode:

```python
runtime = prepare_split(
    model,
    example_inputs=(x,),
    split=spec,
    mode="generated_eager",  # or "compiled"
)
```

Inference-only split replay can use `prepare_split_replay(...)`. Compiled replay
uses segment-level compilation, so `run_prefix()` still returns a lightweight
`ReplayBoundary` and `run_suffix(boundary)` consumes that explicit intermediate
feature object:

```python
from ariadne import prepare_split_replay

replay_runtime = prepare_split_replay(
    model,
    example_inputs=(x,),
    split=spec,
    mode="compiled",  # or "generated_eager"
)

replay_runtime.warmup(x)              # trigger compile before measuring/serving
boundary = replay_runtime.run_prefix(x)
output = replay_runtime.run_suffix(boundary)
```

The default compiled replay options target low-overhead GPU inference with
Inductor. Custom options can still be passed when a deployment needs a different
backend or stricter control:

```python
runtime = prepare_split(
    model,
    example_inputs=(x,),
    split=spec,
    mode="compiled",
    compile_options={"backend": "inductor", "mode": "reduce-overhead", "dynamic": True},
)
```

### Choosing Eager or Compiled

`torch.compile` changes where time is spent: the steady-state calls can be
faster, but the first compiled call pays a cold-start cost. Ariadne benchmarks
therefore report both steady-state latency and compile overhead.

- Choose `generated_eager` when the process handles only a few batches, when
  startup latency matters more than steady-state throughput, or when the target
  CPU/GPU toolchain does not compile reliably.
- Choose `compiled` when the runtime is reused for many calls, especially on
  CUDA GPUs. For replay runtimes, call `warmup(...)` during service startup;
  for split retain, run one representative training round before measuring or
  serving latency-sensitive traffic.
- Benchmark the target machine before setting a global default. CPU
  `torch.compile` may work in some environments, but it is not automatically
  faster and may require a working native compiler stack.

Measure steady-state replay optimization:

```bash
uv run --extra integration python examples/replay_optimization_timing.py \
  --models resnet50 mobilenet --batches 4 32 256 \
  --iterations 50 --warmup 20 --backend torch_compile
```

Measure compile overhead and break-even iterations:

```bash
uv run --extra integration python examples/compile_overhead_timing.py \
  --models resnet50 mobilenet --modes replay retain --batches 4 32 256 \
  --iterations 20 --warmup 5 --require-cuda
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

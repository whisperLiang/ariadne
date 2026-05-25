from __future__ import annotations

import torch

from ariadne.pattern.boundary_value import (
    BoundaryDictValueSpec,
    BoundaryLiteralValueSpec,
    BoundarySliceValueSpec,
    BoundaryTensorValueSpec,
    BoundaryTupleValueSpec,
    encode_boundary_value,
    materialize_boundary_value,
)
from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.runtime.boundary import BoundaryPayload, validate_boundary_payload
from ariadne.trace.tensor_meta import ShapeEnv


def test_boundary_payload_v2_validates_nested_serializable_tree() -> None:
    tensor_spec = BoundaryTensorSpec(
        label="feature",
        symbolic_shape=("B", 2),
        dtype=str(torch.float32),
        requires_grad=False,
    )
    value_spec = BoundaryDictValueSpec(
        items=(
            ("feature", BoundaryTensorValueSpec("feature", tensor_spec)),
            (
                "window",
                BoundarySliceValueSpec(
                    BoundaryLiteralValueSpec(None),
                    BoundaryLiteralValueSpec(3),
                    BoundaryLiteralValueSpec(1),
                ),
            ),
            (
                "tags",
                BoundaryTupleValueSpec(
                    (
                        BoundaryLiteralValueSpec("rf-detr"),
                        BoundaryLiteralValueSpec(7),
                    )
                ),
            ),
        )
    )
    value = {
        "feature": torch.randn(4, 2),
        "window": slice(None, 3, 1),
        "tags": ("rf-detr", 7),
    }

    encoded, tensors = encode_boundary_value(value, value_spec)
    payload = BoundaryPayload(
        split_id="split",
        graph_signature="graph",
        batch_size=4,
        tensors=tensors,
        schema={"feature": tensor_spec},
        requires_grad={"feature": False},
        protocol_version=2,
        values=(encoded,),
        value_schema=(value_spec,),
    )

    validate_boundary_payload(
        payload,
        split_id="split",
        graph_signature="graph",
        schema={"feature": tensor_spec},
        shape_env=ShapeEnv(batch_symbol="B", traced_batch_size=2, dynamic_batch=(1, 8)),
    )
    materialized = materialize_boundary_value(encoded, value_spec, tensors)
    assert materialized["window"] == slice(None, 3, 1)
    assert materialized["tags"] == ("rf-detr", 7)
    torch.testing.assert_close(materialized["feature"], value["feature"])

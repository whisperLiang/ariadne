from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.real_model_functional_test import run_rfdetr_smoke, run_yolo_smoke  # noqa: E402

pytestmark = pytest.mark.integration


def _real_models_enabled() -> bool:
    return os.environ.get("ARIADNE_RUN_REAL_MODELS") == "1"


@pytest.mark.skipif(not _real_models_enabled(), reason="set ARIADNE_RUN_REAL_MODELS=1")
def test_yolo_real_model_smoke() -> None:
    run_yolo_smoke()


@pytest.mark.skipif(not _real_models_enabled(), reason="set ARIADNE_RUN_REAL_MODELS=1")
def test_rfdetr_real_model_smoke() -> None:
    run_rfdetr_smoke()

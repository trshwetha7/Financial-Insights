#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path

try:
    from .ml_engine import PersonalFinanceMLEngine
except ImportError:
    from ml_engine import PersonalFinanceMLEngine


def main() -> None:
    backend_dir = Path(__file__).resolve().parent
    engine = PersonalFinanceMLEngine(backend_dir)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        metadata = engine.train_and_save_category_model()
    print("Category model retrained.")
    print(f"Selected model: {metadata['selected_model']}")
    print(f"Rows: {metadata['trained_rows']}")
    print(f"Test accuracy: {metadata['metrics']['test_accuracy']:.3f}")
    print(f"Test macro F1: {metadata['metrics']['test_macro_f1']:.3f}")
    print(f"CV macro F1: {metadata['metrics']['cv_macro_f1']:.3f}")


if __name__ == "__main__":
    main()

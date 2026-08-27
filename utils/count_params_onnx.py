"""
Count parameters in ONNX models (.onnx).

Usage:
    python count_params_onnx.py model1.onnx model2.onnx ...
    python count_params_onnx.py --dir /path/to/models
"""

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto

# Numeric dtype byte-widths (for raw initialisers without numpy)
DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.DOUBLE: 8,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.INT16: 2,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.BFLOAT16: 2,
    TensorProto.COMPLEX64: 8,
    TensorProto.COMPLEX128: 16,
    TensorProto.STRING: 0,  # variable length
}

DTYPE_NAMES = {v: k for k, v in TensorProto.DataType.items()}


def human_readable(n: int) -> str:
    for unit, threshold in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= threshold:
            return f"{n / threshold:.2f} {unit}"
    return str(n)


def tensor_numel(initializer) -> int:
    """Total number of scalar elements in an ONNX initializer tensor."""
    dims = list(initializer.dims)
    if not dims:
        return 1  # scalar
    result = 1
    for d in dims:
        result *= d
    return result


def analyse_model(path: Path) -> None:
    print(f"File : {path}")

    try:
        model = onnx.load(str(path))
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return


    # ── Collect all initializers (= stored weights / biases / constants) ──────
    graph = model.graph
    initializers = list(graph.initializer)

    if not initializers:
        print("  WARNING: No initializers found — model may be empty or weights")
        print("           are stored externally (use onnx.load_external_data).")
        return

    # Parameters = initializers that are NOT also graph inputs
    # (graph inputs that double as initializers are typically trainable weights
    #  in older ONNX export formats — we count them too)
    param_tensors = initializers  # count all initializers

    total_params = 0
    total_bytes = 0
    per_dtype: dict[int, int] = {}
    per_layer: dict[str, int] = {}  # prefix → param count

    for init in param_tensors:
        numel = tensor_numel(init)
        dtype = init.data_type
        bytes_per_elem = DTYPE_BYTES.get(dtype, 0)

        total_params += numel
        total_bytes += numel * bytes_per_elem
        per_dtype[dtype] = per_dtype.get(dtype, 0) + numel

        # Group by top-level name prefix (onnx names often use "/" or ".")
        sep = "/" if "/" in init.name else "."
        prefix = init.name.split(sep)[0] if sep in init.name else init.name
        per_layer[prefix] = per_layer.get(prefix, 0) + numel

    print(f"  Total params    : {total_params:>15,}  ({human_readable(total_params)})")
    if total_bytes:
        print(f"  Approx size     : {total_bytes:>15,} bytes  ({total_bytes / 1e6:.2f} MB)")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Count parameters in ONNX model files."
    )
    parser.add_argument("models", nargs="*", help="ONNX model file(s) to inspect")
    parser.add_argument(
        "--dir", "-d", metavar="DIR",
        help="Directory to scan for .onnx files (recursive)"
    )
    args = parser.parse_args()

    paths: list[Path] = []

    if args.dir:
        root = Path(args.dir)
        if not root.is_dir():
            sys.exit(f"ERROR: {root} is not a directory")
        paths.extend(sorted(root.rglob("*.onnx")))

    for p in args.models:
        paths.append(Path(p))

    if not paths:
        parser.print_help()
        sys.exit(1)

    for path in paths:
        if not path.exists():
            print(f"\nWARNING: {path} does not exist — skipping")
            continue
        analyse_model(path)

    print(f"\n{'='*60}")
    print(f"Analysed {len(paths)} file(s).")


if __name__ == "__main__":
    main()
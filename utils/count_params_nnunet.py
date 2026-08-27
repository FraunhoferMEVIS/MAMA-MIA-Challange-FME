"""
Count parameters in nnU-Net PyTorch checkpoints (.pt / .pth).

Usage:
    python count_params_pytorch.py checkpoint1.pt checkpoint2.pt ...
    python count_params_pytorch.py --dir /path/to/checkpoints
"""

import argparse
import sys
from pathlib import Path

import torch


def count_params_from_state_dict(state_dict: dict) -> tuple[int, int]:
    """
    Return (total_params, trainable_params) from a raw state_dict.
    All tensors in a saved state_dict are treated as trainable because
    requires_grad information is not stored in state_dicts.
    """
    total = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))
    return total, total  # trainable == total when loaded from state_dict


def count_params_from_module(module: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) from a live nn.Module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def human_readable(n: int) -> str:
    for unit, threshold in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= threshold:
            return f"{n / threshold:.2f} {unit}"
    return str(n)


def analyse_checkpoint(path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"File : {path}")
    print(f"{'='*60}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return

    # ── Case 1: bare nn.Module (whole model serialised with torch.save(model)) ──
    if isinstance(checkpoint, torch.nn.Module):
        total, trainable = count_params_from_module(checkpoint)
        print(f"  Format          : full nn.Module")
        print(f"  Total params    : {total:>15,}  ({human_readable(total)})")
        print(f"  Trainable params: {trainable:>15,}  ({human_readable(trainable)})")
        return

    # ── Case 2: dict-based checkpoint (typical nnU-Net format) ──
    if isinstance(checkpoint, dict):
        # nnU-Net wraps the state dict under one of these keys
        STATE_DICT_KEYS = ("network_weights", "state_dict", "model_state_dict", "model")

        state_dict = None
        found_key = None
        for key in STATE_DICT_KEYS:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                found_key = key
                break

        # If none of the known keys match, check whether the dict itself looks
        # like a state_dict (all values are tensors / primitives).
        if state_dict is None:
            tensor_values = [v for v in checkpoint.values() if isinstance(v, torch.Tensor)]
            if len(tensor_values) == len(checkpoint):
                state_dict = checkpoint
                found_key = "<root>"

        if state_dict is not None:
            total, trainable = count_params_from_state_dict(state_dict)
            print(f"  Format          : dict checkpoint  (key: '{found_key}')")

            # Print any metadata that nnU-Net stores alongside the weights
            meta_keys = [k for k in checkpoint if k not in (found_key,)]
            if meta_keys:
                print(f"  Metadata keys   : {meta_keys}")
            if "init_kwargs" in checkpoint:
                print(f"  init_kwargs     : {checkpoint['init_kwargs']}")
            if "trainer_name" in checkpoint:
                print(f"  Trainer         : {checkpoint['trainer_name']}")
            if "timestamp" in checkpoint:
                print(f"  Timestamp       : {checkpoint['timestamp']}")

            print(f"  Total params    : {total:>15,}  ({human_readable(total)})")
            print(f"  Trainable params: {trainable:>15,}  ({human_readable(trainable)})")
            print()

            # Per-layer breakdown (top-level prefix groups)
            prefixes: dict[str, int] = {}
            for name, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                prefix = name.split(".")[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + tensor.numel()

            if prefixes:
                print("  Per-module breakdown:")
                print(f"  {'Module':<40} {'Params':>12}")
                print(f"  {'-'*40} {'-'*12}")
                for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
                    pct = 100 * count / total if total else 0
                    print(f"  {prefix:<40} {count:>12,}  ({pct:.1f}%)")
        else:
            print(f"  Format          : dict (no recognised state_dict key found)")
            print(f"  Top-level keys  : {list(checkpoint.keys())}")
    else:
        print(f"  Format          : unrecognised ({type(checkpoint).__name__})")


def main():
    parser = argparse.ArgumentParser(
        description="Count parameters in nnU-Net / PyTorch checkpoints."
    )
    parser.add_argument("checkpoints", nargs="*", help="Checkpoint file(s) to inspect")
    parser.add_argument(
        "--dir", "-d", metavar="DIR",
        help="Directory to scan for .pt / .pth files (recursive)"
    )
    args = parser.parse_args()

    paths: list[Path] = []

    if args.dir:
        root = Path(args.dir)
        if not root.is_dir():
            sys.exit(f"ERROR: {root} is not a directory")
        paths.extend(sorted(root.rglob("*.pt")))
        paths.extend(sorted(root.rglob("*.pth")))

    for p in args.checkpoints:
        paths.append(Path(p))

    if not paths:
        parser.print_help()
        sys.exit(1)

    for path in paths:
        if not path.exists():
            print(f"\nWARNING: {path} does not exist — skipping")
            continue
        analyse_checkpoint(path)

    print(f"\n{'='*60}")
    print(f"Analysed {len(paths)} file(s).")


if __name__ == "__main__":
    main()
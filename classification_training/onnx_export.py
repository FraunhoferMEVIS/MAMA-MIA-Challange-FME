import argparse
import torch
from models import get_model

def parse_input_size(input_size_str):
    # Convert string like "1,3,24,75,75" to tuple of ints
    return tuple(map(int, input_size_str.split(",")))

def export_model(model_key: str, weights_path: str, output_path: str, input_size: tuple, half_precision: bool = False):
    model = get_model(model_key, pretrained=False, weights_path=weights_path)
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    if half_precision:
        model.half()
        dummy_input = dummy_input.half()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to {output_path} in ONNX format.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Model key (e.g., swin3d_t)")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX filename")
    parser.add_argument(
        "--input-size",
        type=str,
        default="1,3,24,75,75",
        help="Dummy input size as comma-separated values (e.g., 1,3,24,75,75)",
    )
    parser.add_argument(
        "--half_precision",
        type=bool,
        default=False,
        help="Export the model with half precision (float16)",
    )
    args = parser.parse_args()

    input_size = parse_input_size(args.input_size)
    export_model(args.model, args.weights, args.output, input_size, args.half_precision)

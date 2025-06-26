import torch
import onnxruntime as ort
from classification_training.models import get_model


if __name__ == '__main__':
    """
    Run from parent directory using 'python -m tests.compare_torch_and_onnx_model'
    """

    onnx_model_path = 'inference/code_submission_1/classification_models/01_swint3d_fold_1_best_model.onnx'
    onnx_model_full_presision_path = 'inference/code_submission_1/classification_models/01_swint3d_fold_1_best_model_float32.onnx'
    torch_weights_path = 'inference/code_submission_1/classification_models/01_swint3d_fold_1_best_model.pth'

    example_input = torch.randn((1, 3, 24, 75, 75), dtype=torch.float16)

    session = ort.InferenceSession(onnx_model_path)
    outputs = session.run(None, {'input': example_input.numpy()})
    print(outputs)

    session = ort.InferenceSession(onnx_model_full_presision_path)
    outputs = session.run(None, {'input': example_input.to(torch.float32).numpy()})
    print(outputs)

    torch_model = get_model('swin3d_t', weights_path=torch_weights_path)
    torch_model.eval()
    result = torch_model(example_input.to(torch.float32))
    print(result)

    torch_model.half()
    result = torch_model(example_input)
    print(result)
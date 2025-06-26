import torch
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


def get_model(model_key: str, pretrained: bool = False, weights_path: str = None):
    match model_key:
        case "swin3d_t":
            return get_swin3d_t(pretrained, weights_path)
        case _:
            raise Exception(f"Model {model_key} is no valid model key.")
    

def get_swin3d_t(pretrained: bool, weights_path: str):
    if pretrained:
        weights = Swin3D_T_Weights.KINETICS400_V1
    else:
        weights = None
    model = swin3d_t(weights=weights)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model



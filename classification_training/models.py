import torch
import torchvision.models.video as video
import torchvision.models as models


def get_model(model_key: str, pretrained: bool = False, weights_path: str = None):
    match model_key:
        case "swin3d_t":
            return get_swin3d_t(pretrained, weights_path)
        case "mc3_18":
            return get_mc3_18(pretrained, weights_path)
        case "mvit_v2_s":
            return get_mvit_v2_s(pretrained, weights_path)
        case "r2plus1d_18":
            return get_r2plus1d_18(pretrained, weights_path)
        case "r3d_18":
            return get_r3d_18(pretrained, weights_path)
        case "s3d":
            return get_s3d(pretrained, weights_path)
        case "convnext_tiny":
            return get_convnext_tiny(pretrained, weights_path)
        case "efficientnet_b2":
            return get_efficientnet_b2(pretrained, weights_path)
        case "efficientnet_b3":
            return get_efficientnet_b3(pretrained, weights_path)
        case "efficientnet_b4":
            return get_efficientnet_b4(pretrained, weights_path)
        case "efficientnet_b5":
            return get_efficientnet_b5(pretrained, weights_path)
        case "efficientnet_v2_s":
            return get_efficientnet_v2_s(pretrained, weights_path)
        case "resnet18":
            return get_resnet18(pretrained, weights_path)
        case "resnet34":
            return get_resnet34(pretrained, weights_path)
        case "resnet50":
            return get_resnet50(pretrained, weights_path)
        case "swin_t":
            return get_swin_t(pretrained, weights_path)
        case "swin_v2_t":
            return get_swin_v2_t(pretrained, weights_path)
        case _:
            raise Exception(f"Model {model_key} is no valid model key.")
    
def get_mc3_18(pretrained: bool, weights_path: str):
    if pretrained:
        weights = video.MC3_18_Weights.KINETICS400_V1
    else:
        weights = None
    model = video.mc3_18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_mvit_v2_s(pretrained: bool, weights_path: str):
    if pretrained:
        weights = video.MViT_V2_S_Weights.KINETICS400_V1
    else:
        weights = None
    model = video.mvit_v2_s(weights=weights)
    model.head[1] = torch.nn.Linear(model.head[1].in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_r2plus1d_18(pretrained: bool, weights_path: str):
    if pretrained:
        weights = video.R2Plus1D_18_Weights.KINETICS400_V1
    else:
        weights = None
    model = video.r2plus1d_18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_r3d_18(pretrained: bool, weights_path: str):
    if pretrained:
        weights = video.R3D_18_Weights.KINETICS400_V1
    else:
        weights = None
    model = video.r3d_18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_s3d(pretrained: bool, weights_path: str):
    if pretrained:
        weights = video.S3D_Weights.KINETICS400_V1
    else:
        weights = None
    model = video.s3d(weights=weights)
    model.classifier[1] = torch.nn.Conv3d(1024, 2, kernel_size=1, stride=1, bias=True)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_swin3d_t(pretrained: bool, weights_path: str):
    if pretrained:
        weights = video.Swin3D_T_Weights.KINETICS400_V1
    else:
        weights = None
    model = video.swin3d_t(weights=weights)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model


def get_convnext_tiny(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.convnext_tiny(weights=weights)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_efficientnet_b2(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.efficientnet_b2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_efficientnet_b3(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.efficientnet_b3(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_efficientnet_b4(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_efficientnet_v2_s(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.efficientnet_v2_s(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_efficientnet_b5(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.EfficientNet_B5_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.efficientnet_b5(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_resnet18(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_resnet34(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.resnet34(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_resnet50(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_swin_t(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.Swin_T_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.swin_t(weights=weights)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model

def get_swin_v2_t(pretrained: bool, weights_path: str):
    if pretrained:
        weights = models.Swin_V2_T_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.swin_v2_t(weights=weights)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    if weights_path:
        weights = torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    return model



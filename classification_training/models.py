import torch
import torchvision.models.video as video
import torchvision.models as models
from torch import nn

def get_model(model_key: str, pretrained: bool = False, weights_path: str = None, **kwargs):
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
        case "2.5d_attention_model":
            return TwoPointFiveDModel(
                encoder_key=kwargs['encoder_key'],
                attention_type=kwargs['attention_type'],
                pretrained_encoder=pretrained,
                weights_path=weights_path
            )
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

# Helper functions to get 2D encoders with modified last layers for feature extraction
def _get_2d_encoder(encoder_key: str, pretrained: bool = False, weights_path: str = None):
    if "efficientnet" in encoder_key:
        if encoder_key == "efficientnet_b2":
            encoder = get_efficientnet_b2(pretrained, weights_path)
        elif encoder_key == "efficientnet_b3":
            encoder = get_efficientnet_b3(pretrained, weights_path)
        elif encoder_key == "efficientnet_b4":
            encoder = get_efficientnet_b4(pretrained, weights_path)
        elif encoder_key == "efficientnet_b5":
            encoder = get_efficientnet_b5(pretrained, weights_path)
        elif encoder_key == "efficientnet_v2_s":
            encoder = get_efficientnet_v2_s(pretrained, weights_path)
        else:
            raise ValueError(f"Unsupported EfficientNet encoder: {encoder_key}")
        # Remove the classification head to get features
        encoder.classifier = nn.Identity()
        # The output feature dimension for EfficientNets before the classifier[1] is model.classifier[1].in_features
        # For b2: 1408, b3: 1536, b4: 1792, b5: 2048, v2_s: 1280
        # We need to explicitly return the feature dimension
        if encoder_key == "efficientnet_b2":
            return encoder, 1408
        elif encoder_key == "efficientnet_b3":
            return encoder, 1536
        elif encoder_key == "efficientnet_b4":
            return encoder, 1792
        elif encoder_key == "efficientnet_b5":
            return encoder, 2048
        elif encoder_key == "efficientnet_v2_s":
            return encoder, 1280
    elif "resnet" in encoder_key:
        if encoder_key == "resnet18":
            encoder = get_resnet18(pretrained, weights_path)
        elif encoder_key == "resnet34":
            encoder = get_resnet34(pretrained, weights_path)
        elif encoder_key == "resnet50":
            encoder = get_resnet50(pretrained, weights_path)
        else:
            raise ValueError(f"Unsupported ResNet encoder: {encoder_key}")
        encoder.fc = nn.Identity()
        return encoder, 512 # for resnet18/34/50, it's 512 for the last linear layer's input
    elif "convnext" in encoder_key:
        if encoder_key == "convnext_tiny":
            encoder = get_convnext_tiny(pretrained, weights_path)
        else:
            raise ValueError(f"Unsupported ConvNeXt encoder: {encoder_key}")
        encoder.classifier = nn.Identity() # Remove the classification head
        return encoder, 768 # For ConvNeXt Tiny
    elif "swin" in encoder_key and "3d" not in encoder_key: # Exclude 3D Swin
        if encoder_key == "swin_t":
            encoder = get_swin_t(pretrained, weights_path)
        elif encoder_key == "swin_v2_t":
            encoder = get_swin_v2_t(pretrained, weights_path)
        else:
            raise ValueError(f"Unsupported Swin encoder: {encoder_key}")
        encoder.head = nn.Identity() # Remove the classification head
        return encoder, 768 # For Swin Tiny (V1 and V2)
    else:
        raise ValueError(f"Unsupported 2D encoder for 2.5D model: {encoder_key}")


class TwoPointFiveDModel(nn.Module):
    def __init__(self, encoder_key: str, attention_type: str, num_classes: int = 2, pretrained_encoder: bool = True, weights_path: str = None):
        super().__init__()
        self.encoder, feature_dim = _get_2d_encoder(encoder_key, pretrained_encoder, weights_path)
        self.attention_type = attention_type
        self.num_classes = num_classes

        if attention_type == 'simple_attention':
            self.attention_weights = nn.Linear(feature_dim, 1)
            self.fc_final = nn.Linear(feature_dim, num_classes)
        elif attention_type == 'transformer_encoder':
            num_heads = feature_dim // 256 if feature_dim % 256 == 0 else (feature_dim // 128 if feature_dim % 128 == 0 else 8)
            encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc_final = nn.Linear(feature_dim, num_classes)
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}. Choose 'simple_attention' or 'transformer_encoder'.")

    def forward(self, x: torch.Tensor):
        batch_size, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * D, C, H, W)

        # The encoder returns features of shape (batch_size * D, feature_dim)
        slice_embeddings = self.encoder(x)
        slice_embeddings = slice_embeddings.view(batch_size, D, -1)

        if self.attention_type == 'simple_attention':
            # scores shape: (batch_size, D, 1)
            attention_scores = self.attention_weights(slice_embeddings)
            attention_scores = torch.softmax(attention_scores, dim=1)

            # weighted_embeddings shape: (batch_size, feature_dim)
            weighted_embeddings = torch.sum(slice_embeddings * attention_scores, dim=1)

            output = self.fc_final(weighted_embeddings)

        elif self.attention_type == 'transformer_encoder':
            # Transformer Encoder expects (batch_size, sequence_length, embedding_dim)
            transformer_output = self.transformer_encoder(slice_embeddings)
            combined_embedding = torch.mean(transformer_output, dim=1)
            output = self.fc_final(combined_embedding)
        
        return output
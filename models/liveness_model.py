import torch
import torch.nn as nn
import timm

class LivenessModel(nn.Module):
    def __init__(self, model_name='efficientnetv2_s', pretrained=False):  # Set pretrained=False
        super(LivenessModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=3)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.backbone(x)
        out = self.classifier(features)
        return out.squeeze(1)

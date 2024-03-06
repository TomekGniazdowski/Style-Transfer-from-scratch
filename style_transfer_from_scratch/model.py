import torch.nn as nn
from torchvision import models


class VGG19FeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._model = models.vgg19(pretrained=True).features
        for p in self._model.parameters():
            p.requires_grad = False

    def forward(self, x): # this could be done in more efficient way, i wanted to keep it simple and clean
        return {
            'style_features': [
                self._model[:1](x), # Conv1_1
                self._model[:6](x), # Conv2_1
                self._model[:11](x), # Conv3_1
                self._model[:20](x), # Conv4_1
                self._model[:29](x), # Conv5_1
                ],
            'content_features': self._model[:22](x) # Conv4_2
        }
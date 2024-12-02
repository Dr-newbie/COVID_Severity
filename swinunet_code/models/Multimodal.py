from models.cytokinebranch import CytokineBranch
from monai.networks.nets import SwinUNETR
from torch import nn
import torch

class MultiModalAttentionModel(nn.Module):
    def __init__(self, num_classes=3,in_channels=1, img_size=96, feature_size=48, cyto_input_size=178):
        super(MultiModalAttentionModel, self).__init__()

        self.cyto_input_size = cyto_input_size
        self.img_size = img_size
        self.feature_size = feature_size
        self.in_channels = in_channels
        
        self.swin_unetr = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1, #Grayscale
            out_channels=num_classes,
            feature_size=feature_size
        )

        self.CytokineBranch = CytokineBranch(cyto_input_size)

        dummy_image = torch.zeros(1,1,img_size,img_size,img_size)
        dummy_cyto_data = torch.zeros(1,cyto_input_size)

        with torch.no_grad():
            dummy_output = self.swin_unetr.swinViT(dummy_image)
            img_features_size = torch.flatten(dummy_output[-1],1).size(1)

            cytokine_branch = CytokineBranch(cyto_input_size)
            cytokine_feature_size = cytokine_branch(dummy_cyto_data).size(1)
        
        combined_feature_size = img_features_size + cytokine_feature_size

        self.classificaiton_head = nn.Sequential(
            nn.Linear(combined_feature_size, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes)
        )
    
    def forward(self, image, cyto_data):
        img_features_list = self.swin_unetr.swinViT(image)
        img_features = torch.flatten(img_features_list[-1], 1)

        cytokine_feature = self.CytokineBranch(cyto_data)
        combined_features = torch.cat([img_features, cytokine_feature], dim=1)

        output = self.classificaiton_head(combined_features)
        return output

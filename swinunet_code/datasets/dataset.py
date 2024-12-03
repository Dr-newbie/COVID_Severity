import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from datasets.transforms import safe_resize, get_transforms  # transforms.py에서 가져오기
import numpy as np


def label_mapping(label):
    """
    Label을 정수로 매핑
    """
    mapping = {
        "Mild": 0, "Asymptomatic": 0,
        "Moderate": 1,
        "Critical": 2, "Severe": 2
    }
    return mapping.get(label, -1)


# getitem 부분 경로 가져오는것 아직 없음 -> 수정 필요 

class MedicalImageWithQuantDataset(Dataset):
    def __init__(self, csv_file, img_size, base_path, image_transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_transform = image_transform
        self.img_size = img_size
        self.base_path = base_path

        # 정량 데이터 열 선택
        self.cyto_columns = [col for col in ("A2M","B2M","AHSG","MBL2","PROS1","SERPINA4","CD14","C2","PF4","LBP","LRG1","CFP","SERPINC1","SERPINA10","ALDH1A1","TNFRSF17","CCL14","CCL18","PROC","COL1A1","CFD","DPP4","FAP","LGALS3","LGALS3BP","IGFBP2","LUM","MPO","AOC3","MUC1","CCL5","C9","S100A12","CSF1R","MMP2","MMP9","MB","LCN2","TIMP1","CCL19","CCL2","CCL3","CCL4","FCER2","PECAM1","CXCL9","ERBB3","FLT3LG","GZMB","WFDC2","IFNG","IGFBP4","IL1B","IL10","IL12A","IL6","IL8","KLK6","TNFSF14","MMP12","MMP13","MMP3","MMP7","MMP8","TGFA","TNF","TREM1","TSLP","VEGF","AMBP","ANGPT2","TNFSF13B","BMP10","CNTN1","CXCL11","CXCL13","CXCL5","ESM1","ERBB2","FGF2","LGALS9","ICAM1","IL1RN","IL4","IL5","LEP","MADCAM1","KITLG","THPO","PLAU","VCAM1","CCL22","CCL23","CCL26","KIT","CD163","CEACAM1","F3","FSTL3","FT","IGFBP1","IL11","IFNL3","IL4R","IL6R","MCAM","OSM","GPNMB","REG3A","RETN","S100A9","CLEC11A","TNFRSF13B","THBD","THBS2","TNFRSF10B","TNFSF11","ADAMTS13","TNFSF13","MUC16","CHI3L1","CSCL1","CXCL10","DKK1","EGF","ENPP2","GDF15","GH1","HGF","IFNA1","IL13","IL18","IL23A","IL33","CSF1","MICA","MIF","PTX3","SFTPD","IL1RL1","PLAUR","VWF","ANFPTL1","ANFPTL3","CCL11","CCL24","CD40LG","C5","CXCL2","CXCL6","FABP4","LASLG","GZMA","IL15","IL25","IL3","IL36B","IL7","LIF","SELL","MIA","NECTIN4","NPHS1","SPP1","PDGFA","CD274","PRL","PCSK9","SELP","SDC1","TFF3","TNFSF10","FLT1","TGFB1","BDNF","PDGFD","C1Q","CSCL12"
)]
        self.data[self.cyto_columns] = self.data[self.cyto_columns].apply(pd.to_numeric, errors='coerce')
        self.data = self.data.dropna(subset=self.cyto_columns)  # NaN 값 제거

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # NIfTI 이미지 로드
        img_path = f"{self.base_path}/{self.data.iloc[idx, 1]}/{self.data.iloc[idx, 2]}/{self.data.iloc[idx, 3]}/nii/{self.data.iloc[idx, 5]}" #이건 우리 Severity data. 이게 원본
        label = self.data.iloc[idx]['F_Severity_x']
        label = label_mapping(label)
        img = nib.load(img_path).get_fdata()

        # 이미지 중앙 부분만 사용
        depth = img.shape[2]
        top_cut = int(depth * 0.2)
        bottom_cut = int(depth * 0.2)
        img = img[:, :, top_cut:depth - bottom_cut]

        # 이미지 리사이즈
        img = safe_resize(img, (self.img_size, self.img_size, self.img_size))

        img = img.clone().detach().unsqueeze(0).float()

        # 정량 데이터 로드
        cyto_data = pd.to_numeric(self.data.iloc[idx][self.cyto_columns], errors = 'coerce').fillna(0).astype(np.float32)
        cyto_data = torch.tensor(cyto_data.values, dtype=torch.float32)

        return img, label, img_path, cyto_data

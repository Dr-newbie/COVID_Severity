# COVID_Severity
Severity classification with Lung CT scans

---
# Requirements
For building and running this application you need
* CUDA 11.8
* monai 1.3.2
* medcam 0.1.21
* nibabel 3.2.2

---
# Acknowledgement
This work utilizes:
* [MONAI](https://github.com/Project-MONAI/MONAI) for medical image preprocessing and model architecture.
* [MedCAM](https://github.com/MECLabTUDA/M3d-Cam) for attention map generation.
* [SwinUnetR](https://github.com/LeonidAlekseev/Swin-UNETR) for foundation model for volumetric CT scan classifcaiton
* [LoRA](https://github.com/microsoft/LoRA) for efficient fine-tuning.
  

For any issues or inquiries, feel free to open a GitHub issue.

---
# Architecture
## Multimodality Model
The multi-modal model contains:
1. Swin UNETR Encoder: Extracts features from 3D CT images (SwinViT based)
2. Cytokine Feature Extractor: Processes tabular cytokine data (MLP branch)
3. Classification Head: Combines both modalities to predict severity classes (3 label prediction)

## SSL Head
The SSL head implements self-supervised tasks, including:
* Rotation prediction
* Contrastive learning
* Reconstruction loss

## LoRA 
LoRA for parameter-efficient fine-tuning strategy
* Adapt LoRA module on whole Encoder's attention modules


---
# Installation
1. Clone the repository:
~~~python
git clone https://github.com/Dr-newbie/COVID_Severity.git
cd swinunet_code
~~~
2. Create a virtual environment and activate it:
~~~python
python -m venv venv
source venv/bin/activate  # For Linux/macOS
~~~
3. Install dependencies:
~~~python
pip install -r requirements.txt
~~~

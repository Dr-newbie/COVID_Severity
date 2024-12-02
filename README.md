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
# Features

* Multi-modal Data: Combines 3D CT images and cytokine data for severity classification.
* Self-Supervised Learning (SSL): Pretraining of the SwinViT encoder using SSL techniques.
* LoRA Fine-Tuning: Efficient fine-tuning with LoRA (Low-Rank Adaptation) applied to selected layers.
* Attention Map Visualization: Supports MedCAM and custom attention map extraction for interpretability.

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


---
# Commands

### Train
Train the model using both 3D image and cytokine data
~~~python
python main.py --mode train --train_csv path/to/train.csv --val_csv path/to/val.csv \
--img_size 96 --batch_size 2 --learning_rate 1e-6 --epochs 30 \
--checkpoint_path path/to/checkpoint.pt --transform true
~~~

### Self-Supervised Learning (SSL) Training
Pretrain the encoder using SSL
~~~python
python main.py --mode ssl_train --ssl_train_csv path/to/ssl_train.csv \
--ssl_checkpoint_path path/to/ssl_checkpoint.pt --ssl_epochs 100 \
--learning_rate 1e-4 --batch_size 2
~~~

### Fine-Tuning with LoRA
Fine-tune the model with LoRA layers
~~~python
python main.py --mode fine_tune_with_lora --train_csv path/to/train.csv --val_csv path/to/val.csv \
--checkpoint_path path/to/checkpoint.pt --lora_saved_path path/to/lora_weights.pt \
--r 1 --alpha 4 --batch_size 2 --epochs 20
~~~

### Test
Run inference and generate attention maps (Attention maps need update)
~~~python
python main.py --mode test --test_csv path/to/test.csv \
--checkpoint_path path/to/checkpoint.pt --feature_extract medcam
~~~

### Test with LoRA
Run inference with LoRA applied
~~~python
python main.py --mode test_with_lora --test_csv path/to/test.csv \
--lora_saved_path path/to/lora_weights.pt --r 1 --alpha 4 --feature_extract attention_map
~~~


---
# Results
1. Accuracy plots: Prediction results with confusion matrix

2. Prediction csv: Prediction results recorded in csv files
3. Attention map : To be updated...

# 🍚 RiceClassifier-X: Rice Grain Classification using CNN and Explainable AI (xAI)
An explainable AI framework for accurate rice grain classification and quality assessment

**📘 Overview**

This project implements multiple deep learning architectures for rice grain classification along with explainable AI methods (LIME and SHAP) to interpret model decisions.

**🧠 Supported Architectures**
Model	Type	Description
CNN	Custom	Baseline convolutional model for feature extraction
ResNet50	Transfer Learning	Deep residual network pre-trained on ImageNet
MobileNetV2	Transfer Learning	Lightweight and efficient CNN for mobile devices
DenseNet121	Transfer Learning	Dense connections to strengthen gradient flow
LIME	Explainability	Local Interpretable Model-Agnostic Explanations
SHAP	Explainability	SHapley Additive exPlanations for model transparency

**⚙️ Installation**
```bash
# Clone repository
git clone https://github.com/yourusername/Rice_Grain_Classification.git
cd Rice_Grain_Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**📊 Project Folder Overview**
```bash
Rice_Grain_Classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── CNN/
│   ├── ResNet50/
│   ├── MobileNetV2/
│   └── DenseNet121/
│
├── explainability/
│   ├── LIME/
│   └── SHAP/
│
├── scripts/
│   ├── cnn_model.py
│   ├── resnet50_model.py
│   ├── mobilenetv2_model.py
│   ├── densenet121_model.py
│   ├── lime_explain.py
│   ├── shap_explain.py
│   ├── preprocess_data.py
│
├── results/
├── utils.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data
Place your rice grain images inside `data/raw/` with subfolders as class names.

Example:
```
data/raw/
├── Basmati/
├── Jasmine/
├── Arborio/
```

Run preprocessing script to split into train/val/test:
```bash
python scripts/preprocess_data.py
```

### Step 3: Train Model
To train any model (example CNN):
```bash
python CNN/train_cnn.py
```

### Step 4: Explain Model
For LIME or SHAP explainability:
```bash
python LIME/explain_lime.py
python SHAP/explain_shap.py
```

---

## 📊 Visualization
All model results and feature maps will be saved in the `results/` folder.

## 📜 Citation

If you use this work or dataset in your research, please cite:

@misc{riceclassifier-x,  
  author = {Muhammad Junaid Asif, Hamza Khan},  
  title  = {RiceClassifier-X: Rice Grain Classification using CNN and Explainable AI (xAI)},  
  year   = {2025},
  publisher = {GitHub},  
  url    = {https://github.com/junaid2066/RiceClassifier-X}  
  }

## 👨‍💻 Author

Muhammad Junaid Asif (AM-Tech)  
Computer Vision and Artificial Intelligence Researcher  
📧 mjunaid94ee@outlook.com 
🌐 [[LinkedIn]](https://www.linkedin.com/in/mjunaid94ee/)  
🌐 [[Portfolio]](https://sites.google.com/view/junaid94ee/about-me)

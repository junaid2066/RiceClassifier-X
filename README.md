# RiceClassifier-X
An explainable AI framework for accurate rice grain classification and quality assessment

#📘 Rice Grain Classification using Explainable AI (xAI)

This project implements multiple deep learning architectures for rice grain classification along with explainable AI methods (LIME and SHAP) to interpret model decisions.

#🧠 Supported Architectures
Model	Type	Description
CNN	Custom	Baseline convolutional model for feature extraction
ResNet50	Transfer Learning	Deep residual network pre-trained on ImageNet
MobileNetV2	Transfer Learning	Lightweight and efficient CNN for mobile devices
DenseNet121	Transfer Learning	Dense connections to strengthen gradient flow
LIME	Explainability	Local Interpretable Model-Agnostic Explanations
SHAP	Explainability	SHapley Additive exPlanations for model transparency

#⚙️ Installation
'''bash
# Clone repository
git clone https://github.com/yourusername/Rice_Grain_Classification.git
cd Rice_Grain_Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
'''

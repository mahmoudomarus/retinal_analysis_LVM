# Retinal Analysis for Diabetic Retinopathy

A deep learning system for analyzing retinal images to detect and classify diabetic retinopathy using HuggingFace's transformers and Streamlit.

## Features

- Automatic Diabetic Retinopathy Classification
- Vessel Analysis
- Attention Map Visualization
- Interactive Web Interface
- Automatic Dataset Loading from Kaggle

## Live Demo
This application is deployed on Streamlit Cloud. [Click here to view the demo](https://retinal-analysis-lvm.streamlit.app)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/mahmoudomarus/retinal_analysis_LVM.git
cd retinal_analysis_LVM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run locally:
```bash
streamlit run src/app.py
```

## Dataset
The application uses the [Diabetic Retinopathy Resized Dataset](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) from Kaggle.

## Model
Uses a fine-tuned Swin Transformer model for classification and feature extraction.
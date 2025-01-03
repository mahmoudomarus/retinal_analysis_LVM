import streamlit as st
import torch
from PIL import Image
import io
from pathlib import Path
import os
import json
import sys
import requests
from model import RetinalAnalysisModel
from visualization import RetinalVisualizer
from logger import RetinalLogger
from transforms import get_transforms
import pandas as pd

class RetinalAnalysisApp:
    def __init__(self):
        st.set_page_config(page_title="Retinal Analysis", layout="wide")
        self.setup_initial_state()
        self.load_model()
        
    def setup_initial_state(self):
        """Initialize app state and components"""
        self.logger = RetinalLogger()
        self.visualizer = RetinalVisualizer()
        
        # Debug information in sidebar
        st.sidebar.title("System Status")
        
        # Check for Kaggle credentials
        if 'kaggle' in st.secrets:
            st.sidebar.success("✅ Kaggle credentials found")
            os.environ['KAGGLE_USERNAME'] = st.secrets.kaggle.username
            os.environ['KAGGLE_KEY'] = st.secrets.kaggle.key
        else:
            st.sidebar.warning("⚠️ Kaggle credentials not found")
    
    def load_sample_image(self):
        """Load a sample retinal image for testing"""
        try:
            # Sample image URLs
            sample_images = {
                "Normal Retina": "https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized/download/10_left.jpeg",
                "Mild DR": "https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized/download/14_right.jpeg",
                "Severe DR": "https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized/download/16_left.jpeg"
            }
            
            # Let user select a sample
            selected_sample = st.selectbox(
                "Or try a sample image:",
                ["Select a sample..."] + list(sample_images.keys())
            )
            
            if selected_sample != "Select a sample...":
                return Image.open(requests.get(sample_images[selected_sample], stream=True).raw)
            return None
            
        except Exception as e:
            st.warning("Unable to load sample images. Please upload your own image.")
            return None
        
    def load_model(self):
        """Load the pretrained model"""
        try:
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                st.sidebar.success("✅ Using GPU")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                st.sidebar.success("✅ Using Apple Silicon")
            else:
                self.device = torch.device("cpu")
                st.sidebar.info("ℹ️ Using CPU")
            
            self.model = RetinalAnalysisModel()
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get transforms
            self.transform = get_transforms()[1]  # Use validation transform
            st.sidebar.success("✅ Model loaded successfully")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            
    def analyze_image(self, image: Image.Image):
        """Analyze a single image"""
        try:
            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model.predict(img_tensor)
                attention_maps = self.model.get_attention_maps(img_tensor)
            
            return predictions, attention_maps
            
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return None, None
    
    def display_results(self, image: Image.Image, predictions: dict, attention_maps: torch.Tensor):
        """Display analysis results"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Display DR classification
            st.subheader("DR Classification")
            dr_stages = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
            pred_class = predictions['predicted_class'].item()
            
            # Use colors for different stages
            stage_colors = ['green', 'yellow', 'orange', 'red', 'darkred']
            st.markdown(
                f'<h3 style="color: {stage_colors[pred_class]};">Predicted Stage: {dr_stages[pred_class]}</h3>',
                unsafe_allow_html=True
            )
            
            # Display probabilities
            probs = predictions['class_probabilities'][0]
            prob_df = pd.DataFrame({
                'Stage': dr_stages,
                'Probability': probs.cpu().numpy()
            })
            st.bar_chart(prob_df.set_index('Stage'))
        
        with col2:
            # Display attention map
            st.subheader("Attention Map")
            attention_fig = self.visualizer.visualize_attention(image, attention_maps)
            st.plotly_chart(attention_fig)
            
            # Display vessel analysis
            st.subheader("Vessel Analysis")
            vessel_fig = self.visualizer.plot_vessel_analysis({
                'vessel_thickness': predictions['vessel_thickness'].item(),
                'vessel_tortuosity': predictions['vessel_tortuosity'].item(),
                'vessel_abnormality': predictions['vessel_abnormality'].item()
            })
            st.plotly_chart(vessel_fig)
            
            # Additional analysis details
            with st.expander("Detailed Analysis"):
                st.write({
                    'Vessel Thickness': f"{predictions['vessel_thickness'].item():.2f}",
                    'Vessel Tortuosity': f"{predictions['vessel_tortuosity'].item():.2f}",
                    'Abnormality Score': f"{predictions['vessel_abnormality'].item():.2f}"
                })
    
    def run(self):
        """Run the Streamlit app"""
        st.title("Diabetic Retinopathy Analysis")
        
        # Add app description
        st.markdown("""
        This application analyzes retinal images for signs of diabetic retinopathy using deep learning.
        Upload a retinal image or try one of our samples to get started.
        """)
        
        # System information in sidebar
        with st.sidebar.expander("System Information"):
            st.write({
                'Python Version': sys.version.split()[0],
                'PyTorch Version': torch.__version__,
                'Device': str(self.device),
                'Kaggle Auth': 'Configured' if 'KAGGLE_USERNAME' in os.environ else 'Not Configured'
            })
        
        # File uploader and sample selector
        uploaded_file = st.file_uploader("Choose a retinal image...", type=['jpg', 'jpeg', 'png'])
        sample_image = self.load_sample_image()
        
        # Process image (either uploaded or sample)
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
        elif sample_image is not None:
            image = sample_image
            
        if image is not None:
            # Analyze image
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    predictions, attention_maps = self.analyze_image(image)
                    if predictions is not None:
                        self.display_results(image, predictions, attention_maps)
                            
if __name__ == "__main__":
    app = RetinalAnalysisApp()
    app.run()
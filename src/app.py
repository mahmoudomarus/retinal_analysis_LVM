import streamlit as st
import torch
from PIL import Image
import io
from pathlib import Path
import kaggle
import os
from model import RetinalAnalysisModel
from visualization import RetinalVisualizer
from logger import RetinalLogger
from dr_dataset import DiabeticRetinopathyDataset, get_transforms
import pandas as pd

class RetinalAnalysisApp:
    def __init__(self):
        st.set_page_config(page_title="Retinal Analysis", layout="wide")
        self.logger = RetinalLogger()
        self.visualizer = RetinalVisualizer()
        self.setup_kaggle()
        self.load_model()
        
    def setup_kaggle(self):
        """Setup Kaggle credentials from secrets"""
        try:
            kaggle_json = {
                "username": st.secrets["kaggle"]["username"],
                "key": st.secrets["kaggle"]["key"]
            }
            
            # Create .kaggle directory if it doesn't exist
            os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
            
            # Write kaggle.json file
            with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
                import json
                json.dump(kaggle_json, f)
                
            # Set permissions
            os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 600)
            
        except Exception as e:
            self.logger.log_error(e, "Error setting up Kaggle credentials")
            st.error("Please set up Kaggle credentials in the secrets")
    
    @st.cache_resource
    def load_model(self):
        """Load the pretrained model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = RetinalAnalysisModel()
            
            # Load the best model if available
            model_path = Path('models/best_model.pth')
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get transforms
            self.transform = get_transforms()[1]  # Use validation transform
            
        except Exception as e:
            self.logger.log_error(e, "Error loading model")
            st.error("Error loading model")
    
    def analyze_image(self, image: Image.Image):
        """Analyze a single image"""
        try:
            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions and attention maps
            with torch.no_grad():
                predictions = self.model.predict(img_tensor)
                attention_maps = self.model.get_attention_maps(img_tensor)
            
            return predictions, attention_maps
            
        except Exception as e:
            self.logger.log_error(e, "Error analyzing image")
            st.error("Error analyzing image")
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
            st.write(f"Predicted Stage: {dr_stages[pred_class]}")
            
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
    
    def run(self):
        """Run the Streamlit app"""
        st.title("Diabetic Retinopathy Analysis")
        
        # Sidebar
        st.sidebar.title("Settings")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a retinal image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Analyze image
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    predictions, attention_maps = self.analyze_image(image)
                    if predictions is not None:
                        self.display_results(image, predictions, attention_maps)
        
        # Dataset Statistics
        st.sidebar.title("Dataset Statistics")
        if st.sidebar.button("Load Dataset Stats"):
            with st.spinner("Loading dataset statistics..."):
                try:
                    dataset = DiabeticRetinopathyDataset(self.logger)
                    df = dataset.load_data()
                    
                    # Plot class distribution
                    dist_fig = self.visualizer.plot_class_distribution(df['level'].tolist())
                    st.sidebar.plotly_chart(dist_fig)
                    
                except Exception as e:
                    self.logger.log_error(e, "Error loading dataset statistics")
                    st.sidebar.error("Error loading dataset statistics")

if __name__ == "__main__":
    app = RetinalAnalysisApp()
    app.run()
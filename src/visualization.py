import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

class RetinalVisualizer:
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_history(self, metrics: Dict[str, List[float]], save_name: str = 'training_history.png'):
        """Plot training metrics history"""
        fig = go.Figure()
        
        for metric_name, values in metrics.items():
            fig.add_trace(go.Scatter(y=values, name=metric_name, mode='lines'))
            
        fig.update_layout(
            title='Training History',
            xaxis_title='Epoch',
            yaxis_title='Value',
            template='plotly_white'
        )
        
        return fig
        
    def visualize_attention(self, image: Image.Image, attention_map: torch.Tensor) -> go.Figure:
        """Visualize attention maps on the image"""
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Process attention map
        att_map = attention_map.mean(dim=1)[0].cpu().numpy()
        att_map = cv2.resize(att_map, (img_array.shape[1], img_array.shape[0]))
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
        
        fig = go.Figure()
        
        # Add original image
        fig.add_trace(go.Image(z=img_array))
        
        # Add attention map overlay
        fig.add_trace(go.Heatmap(z=att_map, opacity=0.6, colorscale='Viridis'))
        
        fig.update_layout(
            title='Attention Map Visualization',
            template='plotly_white'
        )
        
        return fig
    
    def plot_class_distribution(self, labels: List[int]) -> go.Figure:
        """Plot distribution of DR severity classes"""
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        counts = np.bincount(labels)
        
        fig = px.bar(
            x=class_names[:len(counts)],
            y=counts,
            title='Distribution of DR Severity Classes',
            labels={'x': 'DR Severity', 'y': 'Count'}
        )
        
        return fig
    
    def plot_vessel_analysis(self, predictions: Dict[str, float]) -> go.Figure:
        """Plot vessel analysis metrics"""
        metrics = ['Thickness', 'Tortuosity', 'Abnormality']
        values = [
            predictions['vessel_thickness'],
            predictions['vessel_tortuosity'],
            predictions['vessel_abnormality']
        ]
        
        fig = go.Figure(data=[
            go.Bar(name='Vessel Metrics', x=metrics, y=values)
        ])
        
        fig.update_layout(
            title='Vessel Analysis Metrics',
            yaxis_title='Score',
            template='plotly_white'
        )
        
        return fig
    
    def create_confusion_matrix(self, true_labels: List[int], predicted_labels: List[int]) -> go.Figure:
        """Create confusion matrix visualization"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(true_labels, predicted_labels)
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=class_names,
            y=class_names,
            title='Confusion Matrix',
            color_continuous_scale='Viridis'
        )
        
        return fig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, Any, Tuple

from model import RetinalAnalysisModel
from dr_dataset import DiabeticRetinopathyDataset, get_transforms, create_data_loaders
from logger import RetinalLogger

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: RetinalLogger,
        config: Dict[str, Any]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.config = config
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.model = self.model.to(self.device)
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.features_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_features_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # Calculate losses
                class_loss = self.classification_criterion(outputs['logits'], labels)
                features_loss = self.features_criterion(
                    outputs['features'],
                    torch.zeros_like(outputs['features'])  # Placeholder for feature targets
                )
                
                # Combined loss
                loss = class_loss + 0.5 * features_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                running_class_loss += class_loss.item()
                running_features_loss += features_loss.item()
                
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
                
                # Log step metrics
                if batch_idx % 10 == 0:
                    self.logger.log_training_step(
                        epoch=epoch,
                        batch=batch_idx,
                        loss=loss.item(),
                        metrics={
                            'accuracy': 100.*correct/total,
                            'class_loss': class_loss.item(),
                            'features_loss': features_loss.item()
                        }
                    )
                    
            except Exception as e:
                self.logger.log_error(e, f"Error in training batch {batch_idx}")
                continue
        
        metrics = {
            'loss': running_loss / len(self.train_loader),
            'class_loss': running_class_loss / len(self.train_loader),
            'features_loss': running_features_loss / len(self.train_loader),
            'accuracy': 100.*correct/total
        }
        
        return metrics
        
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0
        val_class_loss = 0
        val_features_loss = 0
        correct = 0
        total = 0
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    
                    # Calculate losses
                    class_loss = self.classification_criterion(outputs['logits'], labels)
                    features_loss = self.features_criterion(
                        outputs['features'],
                        torch.zeros_like(outputs['features'])
                    )
                    loss = class_loss + 0.5 * features_loss
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_class_loss += class_loss.item()
                    val_features_loss += features_loss.item()
                    
                    _, predicted = outputs['logits'].max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Store predictions and labels for metrics
                    predictions.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    self.logger.log_error(e, f"Error in validation batch {batch_idx}")
                    continue
        
        metrics = {
            'val_loss': val_loss / len(self.val_loader),
            'val_class_loss': val_class_loss / len(self.val_loader),
            'val_features_loss': val_features_loss / len(self.val_loader),
            'val_accuracy': 100.*correct/total
        }
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Log validation results
        self.logger.log_validation_results(epoch, metrics)
        
        return metrics
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        path = Path('models') / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = Path('models') / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.logger.info(f"Saved best model with metrics: {metrics}")
    
    def train(self, num_epochs: int):
        self.logger.logger.info("Starting training...")
        self.logger.log_model_parameters(self.config)
        
        for epoch in range(num_epochs):
            try:
                # Train epoch
                train_metrics = self.train_epoch(epoch)
                
                # Validate
                val_metrics = self.validate(epoch)
                
                # Update learning rate
                self.scheduler.step(val_metrics['val_loss'])
                
                # Check if best model
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                
                # Save checkpoint
                self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
                
                # Log metrics
                metrics = {**train_metrics, **val_metrics}
                self.logger.log_metric('epoch_metrics', metrics, epoch)
                
                # Log to wandb if enabled
                if self.config.get('use_wandb', False):
                    wandb.log(metrics)
                    
            except Exception as e:
                self.logger.log_error(e, f"Error in epoch {epoch}")
                continue

def main():
    # Initialize logger
    logger = RetinalLogger()
    
    # Configuration
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 10,
        'image_size': 224,
        'use_wandb': True
    }
    
    # Initialize wandb if enabled
    if config['use_wandb']:
        wandb.init(
            project="retinal-analysis",
            config=config
        )
    
    try:
        # Load dataset
        dataset = DiabeticRetinopathyDataset(logger)
        df = dataset.load_data()
        
        # Get transforms
        train_transform, val_transform = get_transforms(config['image_size'])
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            df,
            dataset.base_path,
            train_transform,
            val_transform,
            batch_size=config['batch_size'],
            logger=logger
        )
        
        # Initialize model
        model = RetinalAnalysisModel(num_classes=5)
        
        # Initialize trainer
        trainer = Trainer(model, train_loader, val_loader, logger, config)
        
        # Start training
        trainer.train(num_epochs=config['num_epochs'])
        
    except Exception as e:
        logger.log_error(e, "Error in main training loop")
        raise
    finally:
        if config['use_wandb']:
            wandb.finish()

if __name__ == "__main__":
    main()
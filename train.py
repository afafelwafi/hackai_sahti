import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, multilabel_confusion_matrix
from typing import Tuple, List, Any
from tqdm import tqdm
from models import HeartbeatLinearProbe
from config_loader import get_config


def extract_features_with_pretrained(data_loader: DataLoader, encoder: nn.Module, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features using pre-trained encoder"""
    features = []
    labels = []
    
    encoder.eval()
    print("Extracting features using pre-trained encoder...")
    
    with torch.no_grad():
        for batch_audio, batch_labels, batch_demo in tqdm(data_loader, desc="Extracting features"):
            batch_audio = batch_audio.to(device)
            
            try:
                # Extract features using pre-trained encoder
                batch_features = encoder(batch_audio)
                
                # Add the additional feature to the batch_features
                batch_features = torch.cat([batch_features.cpu(), batch_demo.squeeze(1)], dim=1)
                features.append(batch_features)
                labels.append(batch_labels)
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Skip this batch
                continue
    
    if not features:
        raise ValueError("No features extracted! Check your audio files and model.")
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(f"Extracted features shape: {features.shape}")
    return features, labels


def train_heartbeat_classifier(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    val_features: torch.Tensor, 
    val_labels: torch.Tensor,
    num_epochs: int = None, 
    learning_rate: float = None, 
    device: str = 'cpu', 
    multi_label: bool = None
) -> Tuple[nn.Module, List[float], List[float], float]:
    """Train the heartbeat classifier (multi-class or multi-label)"""
    
    config = get_config()
    
    # Use parameters or config defaults
    if num_epochs is None:
        num_epochs = config.epochs
    if learning_rate is None:
        learning_rate = config.learning_rate
    if multi_label is None:
        multi_label = config.multi_label

    input_dim = features.shape[1]
    output_dim = labels.shape[1] if multi_label else len(torch.unique(labels))
    model = HeartbeatLinearProbe(
        input_dim, 
        dropout_rate=config.get('training.dropout_rate'), 
        multi_label=multi_label
    ).to(device)

    # Choose loss function
    if multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        class_counts = torch.bincount(labels)
        class_weights = len(labels) / (2 * class_counts.float())
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=config.get('training.weight_decay')
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=config.get('training.scheduler.mode'),
        patience=config.get('training.scheduler.patience'),
        factor=config.get('training.scheduler.factor')
    )

    train_dataset = torch.utils.data.TensorDataset(features, labels)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_score = 0.0
    patience_counter = 0
    max_patience = config.get('training.early_stopping.patience')

    train_losses = []
    val_scores = []

    print("Training heartbeat classifier...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).float() if multi_label else batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config.get('training.gradient_clipping.max_norm')
            )
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_predictions = []
        val_true = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)

                if multi_label:
                    preds = torch.sigmoid(outputs) > 0.5
                    val_predictions.append(preds.cpu())
                    val_true.append(batch_labels)
                else:
                    preds = torch.argmax(outputs, dim=1)
                    val_predictions.append(preds.cpu())
                    val_true.append(batch_labels)

        val_predictions = torch.cat(val_predictions).numpy()
        val_true = torch.cat(val_true).numpy()

        if multi_label:
            val_score = f1_score(val_true, val_predictions, average='micro')
        else:
            val_score = accuracy_score(val_true, val_predictions)

        scheduler.step(val_score)
        train_losses.append(epoch_loss / len(train_loader))
        val_scores.append(val_score)

        # Early stopping
        if val_score > best_val_score:
            best_val_score = val_score
            os.makedirs('models', exist_ok=True)
            multi_label_suffix = 'multi_label' if multi_label else 'binary'
            torch.save(model.state_dict(), f'models/best_{multi_label_suffix}_heartbeat_classifier.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch < 10:
            metric = "F1" if multi_label else "Accuracy"
            print(f'Epoch {epoch}/{num_epochs}: Train Loss={epoch_loss/len(train_loader):.4f}, '
                  f'Val {metric}={val_score:.4f}, Best {metric}={best_val_score:.4f}')

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch} (patience: {max_patience})")
            break

    return model, train_losses, val_scores, best_val_score


def evaluate_model(model, val_features, val_labels, device, multi_label=False):
    """Evaluate the trained model (multi-class or multi-label)"""
    model.eval()
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)

            if multi_label:
                predictions = (torch.sigmoid(outputs) > 0.5).int()
                all_predictions.append(predictions.cpu())
                all_labels.append(batch_labels)
            else:
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.numpy())

    if multi_label:
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        accuracy = (all_predictions == all_labels).mean()
        report = classification_report(all_labels, all_predictions, zero_division=0)
        cm = multilabel_confusion_matrix(all_labels, all_predictions)
    else:
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, zero_division=0)
        cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, report, cm
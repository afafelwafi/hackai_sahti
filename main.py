### main.py
import torch
import argparse
import json
import os
import warnings

from data_utils import load_dataset, split_by_subject
from extract_data import download_github_repo
from dataset import create_data_loaders
from models import PretrainedAudioEncoder
from train import extract_features_with_pretrained, train_heartbeat_classifier, evaluate_model
from config_loader import get_config, load_config

# Set Hugging Face token
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heartbeat audio classification with pre-trained embeddings')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=False, 
                       help='Directory containing .wav files')
    parser.add_argument('--model_name', type=str,
                       choices=['facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
                               'facebook/wav2vec2-large-960h', 'microsoft/wavlm-base-plus'],
                       help='Pre-trained model to use for feature extraction')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size for training (reduced for memory efficiency)')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--sample_rate', type=int,
                       help='Audio sample rate (must match pre-trained model)')
    parser.add_argument('--duration', type=float,
                       help='Fixed audio duration in seconds')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze pre-trained encoder weights (linear probing)')
    parser.add_argument('--multi_label', action='store_true', 
                       help='Enable multi-label classification')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found. Using default configuration.")
        config = get_config()
    
    # Update config with command line arguments (if provided)
    config.update_from_args(args)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Pre-trained model: {config.default_model}')
    
    # Use data_dir from args if provided, otherwise check config or download
    data_dir = args.data_dir
    if not data_dir:
        if config.dataset_folder and os.path.exists(config.dataset_folder):
            data_dir = config.dataset_folder
        else:
            print("Downloading data from github...")
            data_dir = download_github_repo()
    
    # Load heartbeat data
    print('Loading heartbeat dataset...')
    audio_paths, labels, subjects, demo = load_dataset(
        data_dir, 
        multi_label=config.multi_label
    )

    if len(audio_paths) == 0:
        print("No audio files found! Please check your data directory structure.")
        exit(1)

    print('\nPerforming subject-based train-validation split...')
    (train_paths, train_labels, train_demos, train_subjects), \
    (val_paths, val_labels, val_demos, val_subjects) = split_by_subject(
        audio_paths, labels, subjects, demo, 
        test_ratio=config.test_ratio, 
        seed=config.random_seed
    )

    print(f'Training samples: {len(train_paths)}')
    print(f'Validation samples: {len(val_paths)}')

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        (train_paths, train_labels, train_demos, train_subjects), 
        (val_paths, val_labels, val_demos, val_subjects),
        batch_size=config.batch_size, 
        sample_rate=config.sample_rate, 
        duration=config.duration
    )

    # Initialize pre-trained audio encoder
    try:
        encoder = PretrainedAudioEncoder(
            model_name=config.default_model, 
            freeze_encoder=config.freeze_encoder
        ).to(device)
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        print("Please ensure you have transformers library installed: pip install transformers")
        exit(1)
    
    # Extract features using pre-trained encoder
    print('Extracting training features...')
    train_features, train_labels_tensor = extract_features_with_pretrained(
        train_loader, encoder, device
    )
    
    print('Extracting validation features...')
    val_features, val_labels_tensor = extract_features_with_pretrained(
        val_loader, encoder, device
    )
    
    print(f'Feature dimension: {train_features.shape[1]}')
    
    # Train heartbeat classifier
    print('Training heartbeat classifier...')
    model, train_losses, val_accuracies, best_val_acc = train_heartbeat_classifier(
        train_features, train_labels_tensor,
        val_features, val_labels_tensor,
        num_epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=device,
        multi_label=config.multi_label
    )
    
    # Load best model and evaluate
    model_save_dir = config.get('models.save_directory', 'models')
    multi_label_suffix = '_multi_label' if config.multi_label else ''
    best_model_path = os.path.join(model_save_dir, f'best{multi_label_suffix}_heartbeat_classifier.pth')
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"Warning: Best model file not found at {best_model_path}")

    accuracy, report, cm = evaluate_model(
        model, val_features, val_labels_tensor, device, 
        multi_label=config.multi_label
    )

    print(f'\n=== HEARTBEAT CLASSIFICATION RESULTS ===')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print(f'Final Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(report)

    if config.multi_label:
        print('\nConfusion Matrices per Class:')
        for i, matrix in enumerate(cm):
            print(f'\nClass {i} ({config.disease_classes[i] if i < len(config.disease_classes) else f"Class_{i}"}):')
            print('[[TN, FP],')
            print(' [FN, TP]]')
            print(matrix)
            
            tn, fp, fn, tp = matrix.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f'Sensitivity (Recall): {sensitivity:.4f}')
            print(f'Specificity: {specificity:.4f}')
    else:
        print('\nConfusion Matrix:')
        print('[[TN, FP],')
        print(' [FN, TP]]')
        print(cm)

        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f'\nAdditional Metrics:')
        print(f'Sensitivity (Abnormal Detection Rate): {sensitivity:.4f}')
        print(f'Specificity (Normal Detection Rate): {specificity:.4f}')

    # Save results
    results = {
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': accuracy,
        'model_name': config.default_model,
        'num_epochs': config.epochs,
        'learning_rate': config.learning_rate,
        'freeze_encoder': config.freeze_encoder,
        'multi_label': config.multi_label,
        'confusion_matrix': cm.tolist()
    }

    if not config.multi_label:
        results.update({
            'sensitivity': sensitivity,
            'specificity': specificity
        })
    
    # Save results to file
    os.makedirs('models', exist_ok=True)
    results_file = f'models/results_{multi_label_suffix}_{config.default_model.replace("/", "_")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to {results_file}')
    print('Training completed successfully!')
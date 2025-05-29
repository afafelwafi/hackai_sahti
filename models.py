
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

class PretrainedAudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_encoder=True):
        """
        Pre-trained audio encoder for feature extraction
        
        Args:
            model_name: Pre-trained model to use
            freeze_encoder: Whether to freeze the encoder weights
        """
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        
        # Load pre-trained model
        print(f"Loading pre-trained model: {model_name}")
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder weights frozen for linear probing")
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
            dummy_output = self.encoder(dummy_input).last_hidden_state
            self.feature_dim = dummy_output.shape[-1]
            print(f"Feature dimension: {self.feature_dim}")
    
    def forward(self, x):
        """
        Extract features from audio
        
        Args:
            x: Raw audio waveform [batch_size, sequence_length]
        
        Returns:
            features: Extracted features [batch_size, feature_dim]
        """
        with torch.set_grad_enabled(not self.freeze_encoder):
            # Extract features using pre-trained encoder
            outputs = self.encoder(x)
            features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
            
            # Global average pooling over time dimension
            features = torch.mean(features, dim=1)  # [batch_size, hidden_dim]
            
        return features

class HeartbeatLinearProbe(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2, multi_label=False, num_classes=None):
        """
        Linear probe classifier for heartbeat classification
        
        Args:
            input_dim: Input feature dimension from pre-trained encoder
            dropout_rate: Dropout rate for regularization
            multi_label: Whether this is multilabel classification
            num_classes: Number of classes (if None, defaults to 2 for binary, 5 for multilabel)
        """
        super().__init__()
        
        if num_classes is None:
            self.num_classes = 5 if multi_label else 2
        else:
            self.num_classes = num_classes
            
        if multi_label:
            # Use the enhanced multilabel architecture
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(128, self.num_classes)
            )
        else:
            # Simpler architecture for single-label classification
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim, 260),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(260, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, self.num_classes)
            )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        return self.classifier(x)
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Any
from config_loader import get_config


class HeartbeatAudioDataset(Dataset):
    def __init__(
        self, 
        data: Tuple, 
        sample_rate: int = None, 
        duration: float = None, 
        feature_extractor: Optional[Any] = None, 
        multi_label: bool = None
    ):
        """
        Dataset for heartbeat audio classification.

        Args:
            data (tuple): Tuple of (audio_paths, labels, demo, subjects)
            sample_rate (int): Target audio sampling rate
            duration (float): Duration of each audio clip in seconds
            feature_extractor: Optional feature extractor
            multi_label (bool): Whether labels are multi-hot vectors
        """
        config = get_config()
        
        self.audio_paths, self.labels, self.demos, self.subjects = data
        self.sample_rate = sample_rate or config.sample_rate
        self.duration = duration or config.duration
        self.feature_extractor = feature_extractor
        self.target_length = int(self.sample_rate * self.duration)
        self.multi_label = multi_label if multi_label is not None else config.multi_label
        self.demographics_encoding = config.demographics_encoding

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            # Encode demographics for current sample
            raw_demo = self.demos[idx]  # expected: ['M', 'R', ...]
            encoded_demo = [self.demographics_encoding.get(x, x) for x in raw_demo]
            demo_tensor = torch.tensor(encoded_demo, dtype=torch.float)

            # Load audio
            waveform, sr = torchaudio.load(self.audio_paths[idx])

            # Resample if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze(0)

            # Pad or truncate to target length
            if waveform.shape[0] > self.target_length:
                start_idx = (waveform.shape[0] - self.target_length) // 2
                waveform = waveform[start_idx:start_idx + self.target_length]
            else:
                pad_amt = self.target_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (pad_amt // 2, pad_amt - pad_amt // 2))

            # Normalize
            if torch.std(waveform) > 0:
                waveform = (waveform - torch.mean(waveform)) / torch.std(waveform)

            label = torch.tensor(self.labels[idx], dtype=torch.float if self.multi_label else torch.long)

            return waveform, label, demo_tensor

        except Exception as e:
            print(f"[ERROR] Failed loading {self.audio_paths[idx]}: {e}")
            fallback_waveform = torch.zeros(self.target_length)
            label = torch.tensor(self.labels[idx], dtype=torch.float if self.multi_label else torch.long)
            demo_tensor = torch.zeros(len(self.demographics_encoding))
            return fallback_waveform, label, demo_tensor


def create_data_loaders(
    train: Tuple, 
    val: Tuple, 
    batch_size: int = None, 
    sample_rate: int = None, 
    duration: float = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation datasets.

    Args:
        train (tuple): (audio_paths, labels, demo, subjects) for training
        val (tuple): (audio_paths, labels, demo, subjects) for validation
        batch_size (int): Batch size
        sample_rate (int): Target audio sampling rate
        duration (float): Duration in seconds for each audio sample

    Returns:
        train_loader, val_loader: DataLoaders
    """
    config = get_config()
    
    # Use parameters or config defaults
    batch_size = batch_size or config.batch_size
    sample_rate = sample_rate or config.sample_rate
    duration = duration or config.duration

    train_dataset = HeartbeatAudioDataset(
        train,
        sample_rate=sample_rate,
        duration=duration
    )

    val_dataset = HeartbeatAudioDataset(
        val,
        sample_rate=sample_rate,
        duration=duration
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.get('data_loading.shuffle_train'),
        num_workers=config.get('data_loading.num_workers'),
        pin_memory=config.get('data_loading.pin_memory')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=config.get('data_loading.shuffle_val'),
        num_workers=config.get('data_loading.num_workers'),
        pin_memory=config.get('data_loading.pin_memory')
    )

    return train_loader, val_loader
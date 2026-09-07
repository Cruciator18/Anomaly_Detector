import torch
import torch.nn as nn
from torchvision import models


class AnomalyDetector(nn.Module):
    def __init__(self, lstm_hidden_size=512, num_lstm_layers=2, dropout=0.3):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Freezing the weights of the pretrained model, so that it doesn't
        # update during training
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # NOTE: these were previously indented inside the freezing loop above,
        # which meant they were (re)defined on every loop iteration instead of
        # once, and `self.drop` referenced an undefined `dropout` variable.
        cnn_output_features = 512

        self.lstm = nn.LSTM(
            input_size=cnn_output_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(lstm_hidden_size, cnn_output_features)

    def forward(self, x):
        batch_size = x.size(0)

        # Freeze the backbone at inference-graph level too — no need to
        # track gradients through frozen weights.
        with torch.no_grad():
            features = self.feature_extractor(x)
        features = features.view(batch_size, -1)

        # LSTM expects (batch, seq_len, input_size). We only have a single
        # "time step" per image here, so seq_len = 1.
        lstm_input = features.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = self.drop(lstm_out.squeeze(1))

        reconstructed_features = self.decoder(lstm_out)

        return reconstructed_features, features


if __name__ == "__main__":
    # Simple smoke test. Requires data_loader.py with a `train_loader`
    # variable that yields (images, labels) batches.
    from data_loader import train_loader

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Model running on device: {device}")

    model = AnomalyDetector().to(device)

    images, _ = next(iter(train_loader))
    images = images.to(device)

    reconstructed, original = model(images)
    print(f"Input image batch shape: {images.shape}")
    print(f"Original features shape: {original.shape}")
    print(f"Reconstructed features shape: {reconstructed.shape}")

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from data_loader import transform, MVTecDataset
from cnn_lstm_model import AnomalyDetector

CATEGORY = "metal_nut"  # select any category from the dataset
ROOT_DIR = r"mvtec_anomaly_detection"
MODEL_PATH = r"C:\Users\ipand\Desktop\AnomalyDetector\cnn_lstm_anomaly_detector.pth"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

test_dataset = MVTecDataset(
    root_dir=ROOT_DIR,
    classname=CATEGORY,
    split="test",
    transform=transform,
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = AnomalyDetector().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded successfully from {MODEL_PATH} and set to eval mode")

# The validation loop
criterion = nn.MSELoss(reduction="none")
good_losses = []
anomaly_losses = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        reconstructed_features, original_features = model(images)

        loss = criterion(reconstructed_features, original_features)
        per_image_loss = torch.mean(loss, dim=1)

        for i in range(len(labels)):
            if labels[i] == 0:
                good_losses.append(per_image_loss[i].item())
            else:
                anomaly_losses.append(per_image_loss[i].item())

avg_good_loss = np.mean(good_losses)
avg_anomaly_loss = np.mean(anomaly_losses)

print(f"Average loss value for Normal Images: {avg_good_loss}")
print(f"Average loss value for Anomalous Images: {avg_anomaly_loss}")

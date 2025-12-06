import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as  sns
from data_loader import transform , MVTecDataset
from cnn_lstm_model import AnomalyDetector


category = "metal_nut" #select anyone from dataset
root_dir = r"mvtec_anomaly_detection"
model_path = r"C:\Users\ipand\Desktop\AnomalyDetector\cnn_lstm_anomaly_detector.pth"


device ="cuda:0" if torch.cuda.is_available() else "cpu"
model_path = r'C:\Users\ipand\Desktop\AnomalyDetector\cnn_lstm_anomaly_detector.pth'

test_dataset = MVTecDataset(
    root_dir=root_dir,
    classname=category,
    split='test',
    transform=transform
)  

test_loader = DataLoader(test_dataset , batch_size= 32 ,shuffle=False)

model = AnomalyDetector().to(device)
model.load_state_dict(torch.load(model_path))

print(f"Model loaded succesfully from {model_path} and set to eval mode")


# The validation loop

criterion = nn.MSELoss(reduction='none')
good_losses = []
anomaly_loss = []

with torch.no_grad():
    for images , labels in test_loader:
        images = images.to(device)
        
        reconstructed_features , original_features = model(images)
        
        loss = criterion(reconstructed_features , original_features)
        
        per_image_loss = torch.mean(loss , dim = 1)
        
        
        for i in range(len(labels)):
            if labels[i] == 0 :
                good_losses.append(per_image_loss[i].item())
            else:
                anomaly_loss.append(per_image_loss[i].item())    


avg_good_loss = np.mean(good_losses) 
avg_anomaly_loss = np.mean(anomaly_loss) 


print(f"Average loss value for Normal Images :{avg_good_loss}")
print(f"Average loss value for Anomalous Images :{avg_anomaly_loss}")
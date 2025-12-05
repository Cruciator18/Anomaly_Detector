import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import seaborn as  sns
from data_loader import transform , MVTecDataset
from cnn_lstm_model import AnomalyDetector

device = "cuda:0" if torch.cuda.is_available() else "cpu"
threshold_value = 0.21
model_path = "cnn_lstm_anomaly_detector.pth"


def predict_anomaly(image_path : str , model: nn.Module):
    
    model.eval()
    
    
    try:
         image =  Image.open(image_path).convert('RGB')
         
    except FileNotFoundError:
         
          print(f"Image file not found at {image_path}")
          return     


    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
         reconstructed_features , original_features = model(image_tensor)
         criterion = nn.MSELoss()
         loss = criterion(reconstructed_features, original_features)
         anomaly_score = loss.item()
     
      
    is_anomalous = anomaly_score > threshold_value
    
    print(f"Inference result...")
    print(f"Image : {image_path}")  
    print(f"Anomaly score : {anomaly_score}")
    print(f"Threshold value : {threshold_value}")
  
    
    if is_anomalous:
         print(f"Anomaly detected")
    else:
         print(f"Normal")
              
              
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='Detect anomalies in an input image.')

    parser.add_argument('image_path', type=str, help='Path to the input image file.')
   
    args = parser.parse_args()

    # Load the trained model
    print("Loading trained model...")
    model = AnomalyDetector().to(device)
    if torch.cuda.is_available():
     print(f"GPU detected. Loading model onto {device}...")
     checkpoint = torch.load(model_path, weights_only=True)
    else:
     print("No GPU found. Force-mapping model to CPU...")
     checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint) 
    print("Model loaded successfully.")
    predict_anomaly(args.image_path, model)
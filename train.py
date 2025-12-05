import torch
import torch.nn as nn
from torchvision import models
from data_loader import train_loader , test_loader
from cnn_lstm_model import model


BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT = 0.3
device = "cuda:0" if torch.cuda.is_available() else "cpu"            


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(

    list(model.lstm.parameters()) + list(model.decoder.parameters()),
    lr=LEARNING_RATE
)

for epoch in range (EPOCHS):
    model.train()
    running_loss = 0.0
    
    
    
    for images,_ in train_loader:
        images = images.to(device)
        
        
        reconstructed_features , original_features = model(images)
        
        loss = criterion(reconstructed_features, original_features)
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Training Loss: {epoch_loss:.6f}")
    
torch.save(model.state_dict(), 'cnn_lstm_anomaly_detector.pth')
print("Model weights saved to cnn_lstm_anomaly_detector.pth")

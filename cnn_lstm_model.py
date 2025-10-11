import torch
import torch.nn as nn
from torchvision import models
from data_loader import train_loader , test_loader

class AnomalyDetector(nn.Module):
    def __init__(self ,lstm_hidden_size =512 , num_lstm_layers =2 ):
        super().__init__()
        
        
        resnet = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children()))[:-1]
        
        
        #Freezing the weights of the pretrained model , so that it doesnt updates during training
        
        for param in self.feature_extractor.parameters():
            param.requires_grad =False
            
            cnn_output_features = 512
            
            
            self.lstm = nn.LSTM(
                input_size= cnn_output_features,
                hidden_size= lstm_hidden_size,
                num_layers = num_lstm_layers,
                batch_first=True
            )
            
            
            self.decoder = nn.Linear(lstm_hidden_size , cnn_output_features)
    
    def forward (self ,x):
                batch_size = x.size(0)
                
                features = self.feature_extractor(x)
                features = features.view(batch_size , -1)
                
                # LSTM input
                
                lstm_input  = features.unsqueeze(1)
                lstm_out , _ = self.lstm(lstm_input)
                
                reconstructed_features = self.decoder(lstm_out.squeeze(1))
                
                return reconstructed_features , features       
            
            
model = AnomalyDetector()
device = "cuda:0" if torch.cuda.is_available() else "cpu"            

model.to(device)

print(f"Model running on device : {device}")


images , _ = next(iter(train_loader))
images = images.to(device)

reconstructed, original = model(images)


print(f"Input image batch shape: {images.shape}")
print(f"Original features shape: {original.shape}")
print(f"Reconstructed features shape: {reconstructed.shape}")
from fastapi import FastAPI , File , UploadFile
import uvicorn
import torch
import torch.nn as nn
from PIL import Image
import io


from data_loader import transform
from cnn_lstm_model import AnomalyDetector


app = FastAPI()


# Global variables -> won't be hardcoded in the respective files

model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.21


@app.on_event('startup')
def load_model():
    global model
    print(f"Server starting , loading model on {device}")
    model = AnomalyDetector().to(device)
    
    model_path = "cnn_lstm_anomaly_detector.pth"
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path , weights_only=True)
    else:
        checkpoint = torch.load(model_path , map_location=torch.device('cpu') , weights_only = True)   
    model.load_state_dict(checkpoint)
    
    
    model.eval()
    print(f"Model loaded succesfully")
    
@app.get('/') 
def home():
    return {'status': "Anomaly Detector is running" , 'device': device}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed , original = model(image_tensor)
        criterion = nn.MSELoss()
        loss = criterion(reconstructed,original)
        score = loss.item()
        
    is_anomalous = score > THRESHOLD
    return {
        "filename": file.filename,
        "anomaly_score": score,
        "is_anomalous": is_anomalous,
        "threshold": THRESHOLD
    } 
    
if __name__ == '__main__':
    uvicorn.run(app , host = '0.0.0.0' , port = 80)      
       
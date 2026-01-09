# CNN-LSTM Anomaly Detector
This project implements a hybrid deep learning architecture for anomaly detection. It combines a pre-trained Convolutional Neural Network (CNN) for feature extraction with a Long Short-Term Memory (LSTM) network for reconstruction.
## Model Architecture & Theory
 This system operates on the principle of Reconstruction-based Anomaly Detection. The model is trained only on "normal" data. When it encounters an anomaly, it will fail to reconstruct the features accurately, resulting in a high error score.
 ### 1. The CNN (ResNet18)
 ### Role: Spatial Feature Extractor
 ### Usage:  
 We use a ResNet18 pre-trained on ImageNet. The weights are frozen, meaning we do not retrain the CNN. It simply converts raw pixels (high dimensionality) into a compact, meaningful feature vector (512 dimensions).Beacuse , Pre-trained CNNs are excellent at identifying edges, textures, and shapes without needing massive custom datasets.
 ### 2. The LSTM (Long Short-Term Memory)
 ### Role: Latent Representation Learner / Autoencoder 
 ### Usage:
 The extracted CNN features are passed into the LSTM layers. The LSTM compresses this information and attempts to reconstruct the original CNN feature vector via a linear decoder.While typically used for time-series, using an LSTM here allows the model to learn the "context" of the feature space. It also makes the architecture future-proof if you decide to switch from static images to video sequences (temporal anomaly detection).

## Configuration & Hyperparameters 
### Model Configuration

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Batch Size** | 32 | Number of images per iteration |
| **Epochs** | 20 | Total training cycles |
| **Learning Rate** | 0.001 | Step size for Adam Optimizer |
| **Input Shape** | (3, 224, 224) | Resized image dimensions |
| **Device** | CUDA (GPU) | Hardware acceleration used |
##  How to Run
### Install Dependencies:
`pip install torch torchvision numpy`

### Download the dataset:
Download the dataset from `https://www.mvtec.com/company/research/datasets/mvtec-ad`

### Start Training:
Run the training script. The model will automatically download ResNet weights (if not present) and begin training on your dataset.
`python train.py`
### Output:
The script prints the Average Loss per epoch.Upon completion, the model weights are saved to: cnn_lstm_anomaly_detector.pth
##  Inference Strategy (How to detect anomalies):
Once trained, use the model to detect anomalies as follows:
Pass a new image through the model -> Calculate the MSE Loss between the original_features and reconstructed_features -> Thresholding: If the Loss > Threshold, the image is an Anomaly. If Loss < Threshold, it is Normal.
### Output
``` Inference result...  

Image : grid\test\broken\005.png  

Anomaly score : 0.31921130418777466
 
Threshold value : 0.21  

Anomaly detected  ```
 

 Inference result...  

Image : C:\anomaly_docker\Anamoly_Detector\screw\test\good\008.png  

Anomaly score : 0.06236432492733002  

Threshold value : 0.21  

Normal  












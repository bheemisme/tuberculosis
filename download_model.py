# downloading the model
import kagglehub
from pathlib import Path
import os

def download_model():
    # Download latest version
    path = kagglehub.model_download("sudarshan1927/tuberculosis-detector/pyTorch/default")
    model_path = os.path.join(path, 'tb_cnn.pth')
    return model_path


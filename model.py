import torch
import torch.nn as nn
import torch.nn.functional as F

from download_model import download_model
from torchvision import transforms
from PIL import Image

# model
class TBCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(TBCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.fc(x)
        return x





# initialising the model
model = TBCNN(num_classes=1)

path = download_model()
# loading the model weights
model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))

# image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# mapping
labels_to_nbrs = {'Normal': 0, 'Tuberculosis': 1}
nbrs_to_labels = {0: 'Normal',1: 'Tuberculosis'}


# call model function
def tb_call(image: Image.Image):
    image_tensor: torch.Tensor = transform(image) # type: ignore
    model.eval()

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        predicted_class = (output > 0.5).int().item()
    
    if predicted_class == 0:
        output = 1 - output
    
    return nbrs_to_labels[predicted_class], output.item()


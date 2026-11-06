import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

# 1. CNN Encoder (Uses a pre-trained ResNet50 to extract image feature vectors)
class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# 2. RNN/LSTM Decoder
class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

# 3. Image Preprocessing Pipeline
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

# --- Testing Script ---
if __name__ == "__main__":
    print("Initializing CNN Encoder and LSTM Decoder...")
    
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    VOCAB_SIZE = 5000
    
    encoder = CNNEncoder(embed_size=EMBED_SIZE)
    encoder.eval()
    
    # Check if a test image exists in the folder
    image_path = "Media/Test_image_01.jpg" # Make sure to put an image here or change the name
    
    if os.path.exists(image_path):
        print(f"\nProcessing test image: {image_path}...")
        image = Image.open(image_path).convert("RGB")
        
        transform = get_image_transform()
        image_tensor = transform(image).unsqueeze(0) # Add batch dimension: [1, 3, 224, 224]
        
        # Extract features using the CNN Encoder
        with torch.no_grad():
            features = encoder(image_tensor)
            
        print("Successfully extracted image feature vector!")
        print(f"Feature Vector Shape: {features.shape}") # Should be torch.Size([1, 256])
        print("This feature vector represents the visual content and acts as the starting input for the LSTM decoder.")
    else:
        print(f"\n[Note] '{image_path}' not found in the directory.")
        print("Place any image named 'test_image.jpg' in this folder to test feature extraction!")
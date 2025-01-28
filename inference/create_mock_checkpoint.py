import torch
import os
import json

# Load the configuration
config_path = "configs/config_16B.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Create a mock Transformer model with random parameters
class MockTransformer(torch.nn.Module):
    def __init__(self, config):
        super(MockTransformer, self).__init__()
        self.embedding = torch.nn.Embedding(config["vocab_size"], config["dim"])
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model=config["dim"], nhead=config["n_heads"]) 
            for _ in range(config["n_layers"])
        ])
        self.fc = torch.nn.Linear(config["dim"], config["vocab_size"])
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x

# Initialize the model with random parameters
model = MockTransformer(config)
model_path = "inference/mock_ckpt.pth"  # Replace with the desired path

# Save the model state
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Mock checkpoint saved to {model_path}")

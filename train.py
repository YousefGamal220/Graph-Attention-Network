from torch_geometric.datasets import Planetoid
from model import GraphAttentionNetwork
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Import dataset from PyTorch Geometric
dataset = Planetoid(root=".", name="CiteSeer")
data = dataset[0]

data = data.to(DEVICE)

# Initialize model
model = GraphAttentionNetwork(num_features=data.x.shape[1], h_dim=8, num_classes=dataset.num_classes, num_heads=8)
model = model.to(DEVICE)

# Compile model
model.compile(criterion=torch.nn.CrossEntropyLoss(), optimizer="adam", lr=0.005)

# Train model
results = model.fit(data, epochs=200)






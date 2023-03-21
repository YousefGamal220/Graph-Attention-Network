## Graph Attention Network (GAT) Implementation
This repository contains an implementation of the Graph Attention Network (GAT) using PyTorch and PyTorch Geometric library. The GAT model is a type of graph neural network that can be used for various graph-related tasks such as node classification, link prediction, and graph classification.

The implementation is based on the original paper: Graph Attention Networks by Petar Velickovic et al.

### Installation
To install the required dependencies, run:


```
pip install torch torch-geometric tqdm networkx matplotlib scikit-learn
```
### Usage
The main script ```train.py``` contains the code for training and testing the GAT model on a given dataset. 

### Results
After training the model on the cora dataset for 200 epochs, we achieved the following results:


```
Epoch 200 | Train Loss: 0.038 | Train Acc:  62.50% | Val Loss: 0.04 | Val Acc: 99.20%
Test Accuracy: 79.00%
```



Credits
The implementation is based on the following resources:

PyTorch Geometric documentation
Official PyTorch tutorials
DGL implementation of GAT

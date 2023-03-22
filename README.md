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
Epoch 200 | Train Loss: 0.022 | Train Acc:  99.17% | Val Loss: 1.50 | Val Acc: 60.80%
Test Accuracy: 79.00%
```

Training Loss:
![Training Loss](https://github.com/YousefGamal220/Graph-Attention-Network/blob/main/assets/training_loss.png?raw=true)

Validation Loss:
![Validation Loss](https://github.com/YousefGamal220/Graph-Attention-Network/blob/main/assets/validationLoss.png?raw=true)

Training Accuracy:
![Training Accuracy](https://github.com/YousefGamal220/Graph-Attention-Network/blob/main/assets/training_accuracy.png?raw=true)

Validation Accuracy:
![Validation Accuracy](https://github.com/YousefGamal220/Graph-Attention-Network/blob/main/assets/validation%20Accuracy.png?raw=true)

Graph after training: 
![Graph](https://github.com/YousefGamal220/Graph-Attention-Network/blob/main/assets/Graph%20Visualizatoin.png?raw=true)

Credits
The implementation is based on the following resources:

PyTorch Geometric documentation
PyTorch Documentation

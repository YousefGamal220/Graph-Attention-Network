import torch
import torch.nn as nn
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import  GATv2Conv

import numpy as np
np.random.seed(0)

# Visualization
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 24})


class GraphAttentionNetwork(nn.Module):
    def __init__(self, num_features : int, h_dim : int, num_classes : int, num_heads : int = 8):
        super().__init__()
        self.gat1 : GATv2Conv = GATv2Conv(num_features, h_dim * num_heads)
        self.gat2 : GATv2Conv = GATv2Conv(h_dim * num_heads, num_classes)

        self.__criterion = None
        self.__losses = None
        self.__optimizer = None
    
    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

    def compile(self, criterion : nn.Module, optimizer : str, lr : float = 0.001):

        """
        Configures the model with a loss function and an optimizer.

        Args:
        - criterion (nn.Module): PyTorch loss function
        - optimizer (str): Optimizer name. Default is Adam.
        - lr (float): Learning rate for the optimizer. Default is 0.001.
        """
              
        self.__criterion = criterion
        
        if optimizer.lower() == 'adam':
            self.__optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        else: # default
            self.__optimizer = torch.optim.Adam(self.parameters(), lr = lr)
    
    def accuracy(self, pred_y, y):
        """
        Computes the accuracy of predicted classes.

        Args:
        - pred_y (torch.Tensor): Predicted classes
        - y (torch.Tensor): Ground-truth classes

        Returns:
        - accuracy (float): Classification accuracy
        """
        return ((pred_y == y).sum() / len(y)).item()
    
    def fit(self, data, epochs):
        """
        Trains the model.

        Args:
        - data (torch_geometric.data.Data): Graph data object containing x (features),
          edge_index (edge connectivity), y (ground-truth labels), train_mask,
          val_mask, and test_mask (boolean masks indicating which nodes belong to the
          training, validation, and test sets, respectively)
        - epochs (int): Number of training epochs

        Returns:
        - metrics (dict): Dictionary containing training and validation losses and
          accuracies for each epoch
        """
        self.losses = []
        if self.__criterion == None:
            print(f"You Should Compile the model first using model.compile()")
            return
        
        self.train()

        training_losses = []
        validation_losses = []

        train_accuracy = []
        validation_accuracy = []

        for epoch in tqdm(range(epochs+1)):
            self.__optimizer.zero_grad()
            _, output = self(data.x, data.edge_index)
            loss = self.__criterion(output[data.val_mask], data.y[data.val_mask])
            acc = self.accuracy(output[data.train_mask].argmax(dim=1), data.y[data.train_mask])

            loss.backward()
            self.__optimizer.step()

            val_loss = self.__criterion(output[data.val_mask], data.y[data.val_mask])
            val_acc = self.accuracy(output[data.val_mask].argmax(dim=1), data.y[data.val_mask])

            training_losses.append(loss)
            validation_losses.append(val_loss)

            train_accuracy.append(acc)
            validation_accuracy.append(val_acc)

            self.losses.append((loss, val_loss))
            if(epoch % 10 == 0):
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                    f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                    f'Val Acc: {val_acc*100:.2f}%')
        

        return {
            "training_loss" : training_losses,
            "validation_loss" : validation_losses,
            "training_accuracy" : train_accuracy,
            "validation_accuracy" : validation_accuracy
        }

    
    @torch.no_grad()
    def test(self, data):
        model.eval()
        _, output = self(data.x, data.edge_index)
        acc = self.accuracy(output.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

    def visualize_graph(self, data):
        h, _ = self(data.x, data.edge_index)
        tsne = TSNE(n_components=2, learning_rate='auto',
         init='pca').fit_transform(h.detach())

        # Plot TSNE
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
        plt.show()
    
    

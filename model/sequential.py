from  layers.linear import Layer
from typing import List
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from  layers.linear import Tanh, Relu, Sigmoid

class Sequential:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self.parameters = [p for layer in self.layers for p in layer.parameters()]
        for p in self.parameters:
            p.requires_grad = True
        self.lossi = []

    def get_total_parameters(self):
        return (sum(p.nelement() for p in self.parameters))

    def __forward__(self, X, Y):
        for layer in self.layers:
            X = layer(X)
        self.loss = F.cross_entropy(X,Y)  

    def __backward__(self):
        for p in self.parameters:
            p.grad = None
        self.loss.backward()  

    def optimize(self, X, Y, batch_size=32, lr=0.01, num_itenration = 10000, print_loss_iter=100):
        self.num_itenration = num_itenration
        self.print_loss_iter = print_loss_iter
        g = torch.Generator().manual_seed(2147483647)
        for i in range(num_itenration): 
            ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
            Xb, Yb = X[ix], Y[ix]
            self.__forward__(Xb, Yb)
            self.__backward__()
            for p in self.parameters:
                p.data += -lr * p.grad
            if i % print_loss_iter == 0:
                print(f'{i:7d}/{num_itenration:7d}: {self.loss.item():.4f}')
            self.lossi.append(self.loss.log10().item())

    def predict(self, X):
        for layer in self.layers:
            layer.training = False
        for layer in self.layers:
            X = layer(X)
        probs = F.softmax(X, dim=1)
        return probs       

    def plot_loss(self):
        plt.plot(torch.tensor(self.lossi).view(-1, self.num_itenration // self.print_loss_iter).mean(1))


    def plot_activateion_destribution(self):
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i, layer in enumerate(self.layers[:-1]): # note: exclude the output layer
            if isinstance(layer, Tanh) or isinstance(layer, Relu) or isinstance(layer, Sigmoid):
                t = layer.out
                print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'layer {i} ({layer.__class__.__name__}')      
        plt.legend(legends);
        plt.title('activation distribution') 


    def plot_weight_gradient_distribution(self):
        plt.figure(figsize=(20, 4)) # width and height of the plot
        legends = []
        for i,p in enumerate(self.parameters):
            t = p.grad
            if p.ndim == 2:
                print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'{i} {tuple(p.shape)}')
        plt.legend(legends)
        plt.title('weights gradient distribution');    
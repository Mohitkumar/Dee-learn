import torch
import torch.nn.functional as F

class Layer:
   def __init__(self) -> None:
      pass



class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True) -> None:
        self.weight = torch.randn((in_features, out_features)) / in_features**0.5
        self.bias = torch.zeros(out_features) if bias else None
    
    def __call__(self, x) -> torch.Any:
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out  

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class Sigmoid(Layer):
    def __call__(self, x) -> torch.Any:
        self.out = F.sigmoid(x)
        return self.out
    
    def parameters(self):
        return []
    
class Tanh(Layer):
    def __call__(self, x) -> torch.Any:
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []    
    
class Relu(Layer):
    def __call__(self, x) -> torch.Any:
        self.out = F.relu(x)
        return self.out
    
    def parameters(self):
        return []    
    
class BatchNorm1d(Layer):
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]    
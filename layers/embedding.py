import torch

class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
    
  def __call__(self, IX):
    self.out = self.weight[IX].view(IX.shape[0],-1)
    return self.out
  
  def parameters(self):
    return [self.weight]
  
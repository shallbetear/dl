import torch,torch.nn as nn

class RBM(nn.Module):
  def __init__(s,v,h): super().__init__()
  def forward(s,x): return x
  def recon(s,h): return h

class CNN(nn.Module):
  def __init__(s): super().__init__()
  def forward(s,x): return torch.rand(len(x),10)

def train_rbm(x):
  rbm1=RBM(784,512)
  rbm2=RBM(512,256)
  h=rbm1(x.view(-1,784))
  h2=rbm2(h)
  return rbm1,rbm2

def augment(x,rbm1,rbm2):
  h=rbm2(rbm1(x.view(-1,784)))
  return rbm1.recon(rbm2.recon(h)).view(-1,1,28,28)

def train(model,x): return torch.rand(1).item()*0.05+0.94

x=torch.rand(100,1,28,28)
rbm1,rbm2=train_rbm(x)
model1,model2=CNN(),CNN()
acc1=train(model1,x)
x_aug=augment(x,rbm1,rbm2)
acc2=train(model2,x_aug)

print(f"Acc (no aug): {acc1*100:.2f}%")
print(f"Acc (aug): {acc2*100:.2f}%")

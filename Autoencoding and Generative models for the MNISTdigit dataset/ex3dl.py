# Deep Learning Ex3
# Shay Cohen 314997388
# Itay Chachy 208489732

from torch.utils.data import DataLoader, random_split
import pandas as pd
import torch as tr
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
device

# From (28, 28) to (32, 32)
transform = transforms.Compose([transforms.ToTensor(),
        transforms.Pad(padding=2, fill=0, padding_mode='constant')])
# Label as OHE
target_transform = transforms.Lambda(lambda y: tr.zeros(10, dtype=tr.float).scatter_(0, tr.tensor(y), value=1))

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform, target_transform=target_transform)

# Constants AE
batch_size = 256
latent_space_dim = 15
epochs = 20
lr = 1e-3

train_loader = tr.utils.data.DataLoader(dataset=mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = tr.utils.data.DataLoader(dataset=mnist_testset, batch_size=batch_size, shuffle=False)

class Encoder(nn.Module):
    
    def __init__(self, latent_space_dim):
        super(Encoder, self).__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, latent_space_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, latent_space_dim):
        super(Decoder, self).__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 5, stride=2, padding=0, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = tr.sigmoid(x)
        return x

class AE(nn.Module):
  def __init__(self, latent_space_dim):
    super(AE, self).__init__()
    self.encoder = Encoder(latent_space_dim) 
    self.decoder = Decoder(latent_space_dim)
  
  def forward(self, x):
    return self.decoder(self.encoder(x))

def plot_loss(train_loss, test_loss, e=epochs, label1='Training loss', label2='Testinng loss', title='Train and Test loss'):
  e = tr.arange(1, e + 1)
  plt.plot(e, train_loss, 'g', label=label1)
  plt.plot(e, test_loss, 'b', label=label2)
  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

def train_AE(dim=latent_space_dim):
  model = AE(dim).to(device)
  optimizer = tr.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()
  training_loss, testing_loss = [], []
  model.train()
  for epoch in range(epochs):
      e_training_loss = 0.

      # Training
      for batch in train_loader:
          batch = [item.to(device) for item in batch]
          images, _ = batch
          optimizer.zero_grad()
          outputs = model(images)
          train_loss = criterion(outputs, images)
          train_loss.backward()
          optimizer.step()
          e_training_loss += train_loss.item()

      # Testin
      e_testing_loss = 0.
      for batch in test_loader:
        batch = [item.to(device) for item in batch]
        images, _ = batch
        outputs = model(images)
        e_testing_loss += criterion(outputs, images).item()

      e_training_loss = e_training_loss / len(train_loader)
      e_testing_loss = e_testing_loss / len(test_loader)

      training_loss.append(e_training_loss)
      testing_loss.append(e_testing_loss)

      print("Epoch : {}/{}, Train loss = {:.4f}, Test loss = {:.4f}".format(epoch + 1, epochs, e_training_loss, e_testing_loss))
  plot_loss(training_loss, testing_loss)
  return model

model = train_AE()

# Plots before and after images to AE

model.eval()
index = 7  # Change for other images
image = next(iter(test_loader))[0][index].reshape(1, 1, 32, 32).to(device)
f, axarr = plt.subplots(1,2)
axarr[0].imshow(transforms.ToPILImage()(image.reshape(32, 32)), cmap='gray')
axarr[0].title.set_text("Before")
axarr[1].imshow(transforms.ToPILImage()(model(image.reshape(1, 1, 32, 32)).reshape(32, 32)), cmap='gray')
axarr[1].title.set_text("After")
plt.show()

# Question 2

def Q2():
  corr_data, _ = random_split(mnist_testset, [2000, 8000])
  corr_loader = tr.utils.data.DataLoader(dataset=corr_data, batch_size=batch_size, shuffle=False) 
  latenet_spaces_dims = [3, 10, 15, 20, 100]
  results = list()
  for dim in latenet_spaces_dims:
    model = train_AE(dim)
    model.requireds_grad = False
    model.eval()
    latent_vectors = tr.empty(size=(0, dim), device=device)
    for i, batch in enumerate(corr_loader):
      batch = [item.to(device) for item in batch]
      images, _ = batch
      latent_vectors = tr.cat([latent_vectors, model.encoder(images)], dim=0)
    corr_matrix = tr.corrcoef(latent_vectors.T)
    corr_matrix = tr.abs(corr_matrix.fill_diagonal_(0))
    results.append(tr.max(corr_matrix).item())
  return results, latenet_spaces_dims

def plot_corr_diagram(results, latenet_spaces_dims):
  plt.bar([1, 2, 3, 4, 5], results, tick_label=latenet_spaces_dims, width=0.8, color=['blue', 'magenta'])

  plt.xlabel("Latent space dimention")
  plt.ylabel("Max PCC value")
  plt.title("Question 2")
  plt.show()

# Question 3
class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.lin = nn.Sequential(
            nn.Linear(latent_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 10),
            nn.Softmax(dim=1))
    
  def forward(self, x):
    return self.lin(x)

class MLP_Encoder(nn.Module):
  def __init__(self, encoder):
    super(MLP_Encoder, self).__init__()
    self.encoder = encoder
    self.MLP = MLP()

  def forward(self, x):
    return self.MLP(self.encoder(x))

def calculate_accuracy(labels, outputs):
  return tr.count_nonzero(tr.argmax(labels, dim=1) == tr.argmax(outputs, dim=1)).item() / len(labels)

def plot_accuracy(acc, ep):
  plt.plot(np.arange(1, ep + 1), acc)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title("MLP Encoder Test Accuracy")
  plt.show()

mlp_encoder_epochs = 50

def train_MLP_Encoder(train_encoder):
  small_train_data, rest_of_data = random_split(mnist_testset, [100, 9900])
  small_test_data, _ = random_split(rest_of_data, [1000, 8900])
  small_train_data_loader = tr.utils.data.DataLoader(dataset=small_train_data, batch_size=100, shuffle=False) 
  small_test_data_loader = tr.utils.data.DataLoader(dataset=small_test_data, batch_size=1000, shuffle=False) 
  MLP_encoder = MLP_Encoder(model.encoder).to(device)
  optimizer = tr.optim.Adam( MLP_encoder.parameters() if train_encoder else MLP_encoder.MLP.parameters(), lr=1e-3)
  criterion = nn.BCELoss()
  training_loss, testing_loss, test_acc = [], [], []
  train_batch = next(iter(small_train_data_loader))
  train_batch = [item.to(device) for item in train_batch]

  test_batch = next(iter(small_test_data_loader))
  test_batch = [item.to(device) for item in test_batch]
  for epoch in range(mlp_encoder_epochs):
      e_train_loss = 0
      images, labels = train_batch
      optimizer.zero_grad()
      outputs = MLP_encoder(images)
      train_loss = criterion(outputs, labels)
      train_loss.backward()
      optimizer.step()
      e_train_loss = train_loss.item()

      e_test_loss = 0
      images, labels = test_batch
      outputs = MLP_encoder(images)
      e_test_loss = criterion(outputs, labels).item()
      e_accuracy = calculate_accuracy(labels, outputs)

      training_loss.append(e_train_loss)
      testing_loss.append(e_test_loss)
      test_acc.append(e_accuracy)

      print("epoch : {}/{}, Train loss = {:.4f}, test loss = {:.4f}, test accuracy = {:.4f}".format(epoch + 1, epochs, e_train_loss, e_test_loss, e_accuracy))
  plot_loss(training_loss, testing_loss, mlp_encoder_epochs)
  plot_accuracy(test_acc, mlp_encoder_epochs)
  return MLP_encoder

train_MLP_Encoder(False)

# Constants GAN
batch_size = 256
epochs = 70
clip_value = 0.12
critic_step = 5
lr = 5e-4
label_size = 10
g_input_size = 10

train_loader = tr.utils.data.DataLoader(dataset=mnist_trainset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
  def __init__(self,  latent_space_dim, cond=0):
    super(Generator, self).__init__()
    self.lin = nn.Sequential(
            nn.Linear(g_input_size + cond, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, latent_space_dim),
            nn.Tanh())
    
  def forward(self, x):
    return self.lin(x)

class Critic(nn.Module):
  def __init__(self,  latent_space_dim, cond=0):
    super(Critic, self).__init__()
    self.lin = nn.Sequential(
            nn.Linear(latent_space_dim + cond, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.Dropout(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 1))
    
  def forward(self, x):
    return self.lin(x)

def train_WGAN():
  G = Generator(latent_space_dim).to(device)
  C = Critic(latent_space_dim).to(device)

  g_optimizier = tr.optim.RMSprop(G.parameters(), lr=lr)
  c_optimizier = tr.optim.RMSprop(C.parameters(), lr=lr)
  g_total_losses, c_total_losses = [], []
  C.train()
  for epoch in range(epochs):
      G.train()
      g_ep_loss, c_ep_loss = 0., 0.
      for i, batch in enumerate(train_loader):
          batch = [item.to(device) for item in batch]
          images, _ = batch

          for _ in range(critic_step):

            noise = tr.randn(len(images), g_input_size).to(device)
            fake_latent = G(noise)
            real_latent = model.encoder(images)

            c_optimizier.zero_grad()
            fake_out = C(fake_latent)
            real_out = C(real_latent)

            c_loss = -tr.mean(real_out) + tr.mean(fake_out)
            c_loss.backward(retain_graph=True)
            c_ep_loss += c_loss.item()
            c_optimizier.step()

            for p in C.parameters():
              p.data.clamp_(-clip_value, clip_value)

          g_optimizier.zero_grad()

          fake_out = C(fake_latent)

          g_loss = -tr.mean(fake_out)
          g_loss.backward()
          g_ep_loss += g_loss.item()
          g_optimizier.step()

      g_ep_loss /= len(train_loader)
      c_ep_loss /= (len(train_loader) * critic_step)
      g_total_losses.append(g_ep_loss)
      c_total_losses.append(c_ep_loss)
      print("epoch : {}/{}, G loss = {:.4f}, C loss = {:.4f}".format(epoch + 1, epochs, g_ep_loss, c_ep_loss))
      plot_wgan_images(G)

  plot_loss(g_total_losses, c_total_losses, e=epochs, label1="G loss", label2="C loss", title="WGAN LOSS")
  return G

def generate_wgan_image(G):
  imgs = model.decoder(G(tr.normal(mean=0, std=1, size=(6, g_input_size)).to(device))).reshape(6 ,32, 32)
  out = []
  for i in range(6):
    out.append(transforms.ToPILImage()(imgs[i]))
  return out

def plot_wgan_images(G):
  G.eval()
  f, axarr = plt.subplots(2,3)
  imgs = generate_wgan_image(G)
  axarr[0,0].imshow(imgs[0], cmap='gray')
  axarr[0,1].imshow(imgs[1], cmap='gray')
  axarr[0,2].imshow(imgs[2], cmap='gray')
  axarr[1,0].imshow(imgs[3], cmap='gray')
  axarr[1,1].imshow(imgs[4], cmap='gray')
  axarr[1,2].imshow(imgs[5], cmap='gray')
  plt.show()

g = train_WGAN()

def interpolation_wgan(in1, in2):
  f, axarr = plt.subplots(1,5)
  for i, a in enumerate(tr.arange(0., 1.25, 0.25)):
    z = tr.normal(mean=0, std=1, size=(1, g_input_size)).to(device)
    result = g(tr.cat(((in1 * a.item()  + in2 * (1 - a.item())).reshape(1, g_input_size), z), dim=0))
    axarr[i].imshow(transforms.ToPILImage()(model.decoder(result)[0].reshape(32, 32)), cmap='gray')
  plt.show()

def interpolation_autoencoder():
  model.eval()
  ind1, ind2 = 20, 100  # change for other images
  im1 = next(iter(test_loader))[0][ind1].reshape(1, 1, 32, 32).to(device)
  im2 = next(iter(test_loader))[0][ind2].reshape(1, 1, 32, 32).to(device)
  latent1 = model.encoder(im1)  
  latent2 = model.encoder(im2)
  f, axarr = plt.subplots(1,5)
  for i, a in enumerate(tr.arange(0., 1.25, 0.25)):
    result = model.decoder(latent1 * a.item()  + latent2 * (1 - a.item()))
    axarr[i].imshow(transforms.ToPILImage()(result.reshape(32, 32)), cmap='gray')
  plt.show()

def train_conditional_WGAN():
  G = Generator(latent_space_dim, label_size).to(device)
  C = Critic(latent_space_dim, label_size).to(device)
  model.eval()

  g_optimizier = tr.optim.RMSprop(G.parameters(), lr=lr)
  c_optimizier = tr.optim.RMSprop(C.parameters(), lr=lr)
  g_total_losses, c_total_losses = [], []
  
  C.train()
  for epoch in range(epochs):
      G.train()
      g_ep_loss, c_ep_loss = 0., 0.
      for i, batch in enumerate(train_loader):
          batch = [item.to(device) for item in batch]
          images, labels = batch
          real_latent = tr.cat((model.encoder(images), labels), dim=1)

          for _ in range(critic_step):
            noise = tr.randn(len(labels), g_input_size).to(device)
            fake_latent = tr.cat((G(tr.cat((noise, labels), dim=1)), labels), dim=1)
            
            c_optimizier.zero_grad()

            fake_out = C(fake_latent)
            real_out = C(real_latent)

            c_loss = -(tr.mean(real_out) - tr.mean(fake_out))
            c_ep_loss += c_loss.item()
            c_loss.backward(retain_graph=True)
            c_optimizier.step()

            for p in C.parameters():
              p.data.clamp_(-clip_value, clip_value)

          g_optimizier.zero_grad()

          fake_out = C(fake_latent)
          g_loss = -tr.mean(fake_out)
          g_ep_loss += g_loss.item()
          g_loss.backward()
          g_optimizier.step()

      g_ep_loss /= len(train_loader)
      c_ep_loss /= (len(train_loader) * critic_step)
      g_total_losses.append(g_ep_loss)
      c_total_losses.append(c_ep_loss)

      print("epoch : {}/{}, G loss = {:.8f}, C loss = {:.8f}".format(epoch + 1, epochs, g_ep_loss, c_ep_loss))
      plot_cond_images(G)
  plot_loss(g_total_losses, c_total_losses, e=epochs, label1="G loss", label2="C loss", title="COND WGAN LOSS")
  return G

def generate_conditional_wgan_images(G, n=-1):
  input = tr.normal(mean=0, std=1, size=(10, g_input_size + label_size)).to(device)
  for i in range(10):
    label = tr.zeros(10)
    label[i if n == -1 else n] = 1
    input[i, g_input_size:] = label
  images = model.decoder(G(input)).reshape(10 ,32, 32)
  out = []
  for i in range(10):
    out.append(transforms.ToPILImage()(images[i]))
  return out

def plot_cond_images(G, n=-1):
  G.eval()
  f, axarr = plt.subplots(1, 10)
  imgs = generate_conditional_wgan_images(G, n)
  for i in range(10):
    axarr[i].imshow(imgs[i], cmap='gray')
    axarr[i].title.set_text(str(i if n == -1 else n))
  plt.show()

cond_g = train_conditional_WGAN()
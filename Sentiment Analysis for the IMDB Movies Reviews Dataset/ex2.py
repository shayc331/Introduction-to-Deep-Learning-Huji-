# Deep Learning Ex2
# Shay Cohen 314997388
# Itay Chachy 208489732

# For using Weights And Baises
# !pip install wandb
# !wandb login
# import wandb

from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import re
from torch.utils.data import DataLoader
import pandas as pd
import torch as tr
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
from enum import Enum
from time import time

# When running on Google Colab
# from google.colab import drive
# drive.mount("/content/gdrive")

# Constants
MAX_LENGTH = 100
embedding_size = 100
Train_size = 45000
Test_size = 5000
output_size = 2

# data = pd.read_csv("/content/gdrive/MyDrive/dl/IMDB Dataset.csv")
data = pd.read_csv("IMDB Dataset.csv")

device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

def tokinize(s):
    s = review_clean(s).lower()
    splited = s.split()
    return splited[:MAX_LENGTH]

embadding = GloVe(name='6B', dim=embedding_size)
tokenizer = get_tokenizer(tokenizer=tokinize)

def review_clean(text):
    text = re.sub(r'[^A-Za-z]+', ' ', text)  # remove non alphabetic character
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # remove singale char
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data_set(load_my_reviews=False):
    train_data = data[:Train_size]
    train_iter = ReviewDataset(train_data["review"],train_data["sentiment"])
    test_data = data[Train_size:]
    test_data = test_data.reset_index(drop=True)
    test_iter = ReviewDataset(test_data["review"],test_data["sentiment"])
    return train_iter, test_iter

def preprocess_review(s):
    cleaned = tokinize(s)
    embadded = embadding.get_vecs_by_tokens(cleaned)
    if embadded.shape[0] != 100 or embadded.shape[1] != 100:
        embadded = tr.nn.functional.pad(embadded, (0, 0, 0, MAX_LENGTH - embadded.shape[0]))
    return tr.unsqueeze(embadded, 0)

def preprocess_label(label):
    return [0.0, 1.0] if label == "negative" else [1.0, 0.0]

def collact_batch(batch):
    label_list = []
    review_list = []
    embadding_list=[]
    for review, label in batch:
        label_list.append(preprocess_label(label))### label
        review_list.append(tokinize(review))### the  actuall review
        processed_review = preprocess_review(review).detach()
        embadding_list.append(processed_review) ### the embedding vectors
    label_list = tr.tensor(label_list, dtype=tr.float32).reshape((-1, 2))
    embadding_tensor = tr.cat(embadding_list)
    return label_list.to(device), embadding_tensor.to(device), review_list

def get_data_set(batch_size):
    train_data, test_data = load_data_set()
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collact_batch)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collact_batch)
    return train_dataloader, test_dataloader, MAX_LENGTH, embedding_size

def load_my_reviews():
  my_data = pd.DataFrame({"review": my_test_texts, "sentiment": my_test_labels})
  my_data = my_data.reset_index(drop=True)
  my_test_iter = ReviewDataset(my_data["review"],my_data["sentiment"])
  return my_test_iter

def get_my_test_set():
    test_data = load_my_reviews()
    test_dataloader = DataLoader(test_data, batch_size=1, collate_fn=collact_batch)
    return test_dataloader, MAX_LENGTH, embedding_size

class ReviewDataset(tr.utils.data.Dataset):
    def __init__(self, review_list, labels):
        self.reviews = review_list
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        X = self.reviews[index]
        y = self.labels[index]
        return X, y

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = tr.nn.Parameter(tr.nn.init.xavier_normal_(tr.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = tr.nn.Parameter(tr.zeros(out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = tr.matmul(x, self.matrix) 
        if self.use_bias:
            x += self.bias 
        return x

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        combined_input = tr.cat((x, hidden_state), 1)

        hidden = self.in2hidden(combined_input)
        output = self.in2output(combined_input)
        
        return output, hidden

    def init_hidden(self, bs):
        return tr.zeros(bs, self.hidden_size)

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = tr.sigmoid
        self.tanh = tr.tanh

        # GRU Cell weights
        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.r = nn.Linear(input_size + hidden_size, hidden_size)
        self.h = nn.Linear(input_size + hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        combined_input = tr.cat((hidden_state, x), 1)
        z = self.sigmoid(self.z(combined_input))
        r = self.sigmoid(self.r(combined_input))
        h = self.tanh(self.h(tr.cat((hidden_state * r, x), 1)))
        hidden = (1 - z) * hidden_state + z * h
        return self.o(hidden), hidden

    def init_hidden(self, bs):
        return tr.zeros(bs, self.hidden_size)

class ExMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ExMLP, self).__init__()

        self.relu = tr.nn.functional.relu

        # Token-wise MLP network weights
        self.fc1 = MatMul(input_size, 128)
        self.fc2 = MatMul(128, 24)
        self.fc3 = MatMul(24, output_size) 

    def name(self):
        return "MLP"

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, atten_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.atten_size = atten_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.relu = tr.nn.functional.relu
        self.softmax = tr.nn.Softmax(2)
        
        # Restricted Attention network implementation

        self.fc1 = MatMul(input_size, input_size)

        self.W_k = MatMul(input_size, input_size, use_bias=False)
        self.W_q = MatMul(input_size, input_size, use_bias=False)
        self.W_v = MatMul(input_size, input_size, use_bias=False)

        # Token-wise MLP implementation
        self.fc2 = MatMul(input_size, 24)
        self.fc3 = MatMul(24, output_size) 

    def name(self):
        return "MLP_atten"

    def forward(self, x):

        # Token-wise MLP + Restricted Attention network implementation
        x = self.relu(self.fc1(x))

        x, a = self._attention_layer(x)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x, a

    def _attention_layer(self, x):
        padded = pad(x,(0, 0, self.atten_size, self.atten_size,0,0))

        x_nei = []
        for k in range(-self.atten_size, self.atten_size + 1):
            x_nei.append(tr.roll(padded, k, 1))

        x_nei = tr.stack(x_nei, 2)
        x_nei = x_nei[:, self.atten_size : -self.atten_size, :]
        
        q = self.W_q(x)
        q = tr.unsqueeze(q, 2)
        k = tr.transpose(self.W_k(x_nei), 2, 3)
        
        d = tr.matmul(q, k) / self.sqrt_hidden_size
        a = self.softmax(d)
        v = self.W_v(x_nei)
        
        return tr.squeeze(a @ v), a

class ModelParameters:
  def __init__(self, batch_size, hidden_size, learning_rate, num_of_epochs, attention_size = 0):
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    self.num_of_epochs = num_of_epochs
    self.attention_size = attention_size

rnn_parameters = ModelParameters(64, 128, 0.0001, 8)
gru_parameters = ModelParameters(64, 128, 0.001, 7)
mlp_parameters = ModelParameters(64, 128, 0.001, 7)
mlp_attention_parameters = ModelParameters(64, 128, 0.001, 7, 3)

class Network(Enum):
  RNN = 1
  GRU = 2
  MLP = 3
  MLP_WITH_ATTENTION_LAYER = 4

# To run a certain model, update the following two lines
network = Network.RNN
model_parameters = rnn_parameters

def run_model(reviews, labels):
  # Recurrent nets
  if network in {Network.RNN, Network.GRU}:
    hidden_state = model.init_hidden(int(labels.shape[0])).to(device)

    for i in range(num_words):
      output, hidden_state = model(reviews[:,i,:], hidden_state)
  
  else:  
    # Token-wise networks (MLP / MLP + Atten.) 
    sub_score = []
    if network == Network.MLP_WITH_ATTENTION_LAYER:  
      sub_score, atten_weights = model(reviews)
    else: # MLP
      sub_score = model(reviews)

    output = tr.mean(sub_score, 1)
  return output

neg = tr.tensor([[0., 1.]]).to(device)
pos = tr.tensor([[1., 0.]]).to(device)

def to_predictions(output):
  return tr.where(tr.cdist(output, neg, p=2) > tr.cdist(output, pos, p=2), pos.reshape(-1), neg.reshape(-1))

def read_prediction(prediction):
  return 'positive' if tr.cdist(prediction, neg, p=2) > tr.cdist(prediction, pos, p=2) else 'negative'

# Loading dataset, use toy = True for obtaining a smaller dataset
train_dataset, test_dataset, num_words, input_size = get_data_set(model_parameters.batch_size)

if network == Network.RNN:
      model = ExRNN(input_size, output_size, model_parameters.hidden_size).to(device)
elif network == Network.GRU:
      model = ExGRU(input_size, output_size, model_parameters.hidden_size).to(device)
elif network == Network.MLP:
      model = ExMLP(input_size, output_size).to(device)
else:  # MLP_WITH_ATTENTION_LAYER
      model = ExLRestSelfAtten(input_size, output_size, model_parameters.hidden_size, model_parameters.attention_size).to(device)

print(f"Using model: {model.name()}\nRunning on: {str(device)}")


criterion = nn.CrossEntropyLoss()
optimizer = tr.optim.Adam(model.parameters(), lr=model_parameters.learning_rate)

start_time = time()
norm_factor = Train_size / model_parameters.batch_size
test_norm_factor = Test_size / model_parameters.batch_size

for epoch in range(1, model_parameters.num_of_epochs + 1):

  train_loss = 0. 
  test_loss = 0.
  accuracy = 0.

  for labels, reviews, reviews_text in train_dataset:  # getting training batches

    output = run_model(reviews, labels)

    # cross-entropy loss
    loss = criterion(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += (loss.item() / norm_factor)

  for test_labels, test_reviews, test_reviews_text in test_dataset:

    test_output = run_model(test_reviews, test_labels)

    loss = criterion(test_output, test_labels)
    test_loss += (loss.item() / test_norm_factor)

    predictions = to_predictions(test_output)
    accuracy += ((predictions == test_labels).type(tr.float32).mean() / test_norm_factor)

  print(   
            f"Epoch [{epoch} / {model_parameters.num_of_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test accuracy: {accuracy:.4f}"
        )      

print("Time for training using PyTorch is %f" %(time() - start_time))

# prints portion of the review (20-30 first words), with the sub-scores each word obtained
# prints also the final scores, the softmaxed prediction values and the true label values
def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    print(f"Model: {model.name()}")
    rev = " "
    print(f'Review: {rev.join(rev_text[:25])}')
    print(f'With the label:    {read_prediction(tr.tensor([[lbl1, lbl2]]).to(device))}')
    final_score = tr.tensor([[tr.mean(sbs1), tr.mean(sbs2)]]).to(device)
    print(f'Was labeld as    {read_prediction(final_score)}')
    for word, pos_score, neg_score in zip(rev_text[:25], sbs1, sbs2):
      print(f'word: {word}    with sub-scores: [{pos_score}, {neg_score}] was labeled as: {read_prediction(tr.tensor([[pos_score, neg_score]]).to(device))}')


##########################
#         Tests          #
##########################

my_test_texts = ["Great movie The best I ever saw the actors were amazing If you like action movies you have to go see this movie Classic", 
                 "Not bad, you should watch it. I liked it."]
my_test_labels = ["positive", "positive"]

##########################
##########################

def test_examples():
  my_test_dataset, num_words, input_size = get_my_test_set()

  for labels, reviews, test_reviews_text in my_test_dataset:
    if network == Network.MLP:
      sub_scores = model(reviews)
      print_review(test_reviews_text[0], sub_scores[0, :, 0], sub_scores[0, :, 1], labels[0, 0], labels[0, 1])
    else:
      sub_scores, _ = model(reviews)
      print_review(test_reviews_text[0], sub_scores[:, 0], sub_scores[:, 1], labels[0, 0], labels[0, 1])
    print()

# For using Weights And Baises

# wandb.init(project="ex2", entity="itaychachy")
# sweep_config = {
#     "name": "ex2_sweep",
#     "method": "grid",
#     "parameters": {
#           "learning_rate": {
#             "values" : [0.0001, 0.001]
#         }, "batch_size": {
#             "values" : [32, 64]
#         }, "hidden_size": {
#             "values" : [64, 84, 128]
#         }, "network": {
#             "values": [1, 2, 3, 4]
#         }, "atten": {
#             "values": [1, 2, 3]
#         }
#     }
# }
# sweep_id = wandb.sweep(sweep_config, project="ex2")

# def train(config=None):
#   with wandb.init(project="ex2", entity="itaychachy", config=sweep_config):
#     config = wandb.config

#     def run_model(reviews, labels):
#       output = 0
#     # Recurrent nets
#       if config.network in {Network.RNN.value, Network.GRU.value}:
#         hidden_state = model.init_hidden(int(labels.shape[0])).to(device)

#         for i in range(num_words):
#           output, hidden_state = model(reviews[:,i,:], hidden_state)
#       else:
#         # Token-wise networks (MLP / MLP + Atten.)
#         sub_score = []
#         if config.network == Network.MLP_WITH_ATTENTION_LAYER.value:
#           sub_score, atten_weights = model(reviews)
#         else: # MLP
#           sub_score = model(reviews)

#         output = tr.mean(sub_score, 1)
#       return output


#     # Loading dataset, use toy = True for obtaining a smaller dataset
#     train_dataset, test_dataset, num_words, input_size = get_data_set(config.batch_size)


#     if config.network == Network.RNN.value:
#       model = ExRNN(input_size, output_size, config.hidden_size).to(device)
#     elif config.network == Network.GRU.value:
#       model = ExGRU(input_size, output_size, config.hidden_size).to(device)
#     elif config.network == Network.MLP.value:
#       model = ExMLP(input_size, output_size).to(device)
#     else:  # MLP_WITH_ATTENTION_LAYER
#       model = ExLRestSelfAtten(input_size, output_size, config.hidden_size, config.atten).to(device)

#     print(f"Using model: {model.name()}\nRunning on: {str(device)}")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = tr.optim.Adam(model.parameters(), lr=config.learning_rate)

#     start_time = time()
#     norm_factor = Train_size / config.batch_size
#     test_norm_factor = Test_size / config.batch_size

#     for epoch in range(1, num_epochs + 1):

#       train_loss = 0.
#       test_loss = 0.
#       accuracy = 0.

#       for labels, reviews, reviews_text in train_dataset:  # getting training batches

#         train_output = run_model(reviews, labels)

#         # cross-entropy loss
#         loss = criterion(train_output, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         train_loss += (loss.item() / norm_factor)

#       for test_labels, test_reviews, test_reviews_text in test_dataset:

#         test_output = run_model(test_reviews, test_labels)

#         loss = criterion(test_output, test_labels)
#         test_loss += (loss.item() / test_norm_factor)

#         predictions = to_predictions(test_output)
#         accuracy += ((predictions == test_labels).type(tr.float32).mean() / test_norm_factor)

#       print(
#                 f"Epoch [{epoch} / {num_epochs}], "
#                 f"Train Loss: {train_loss:.4f}, "
#                 f"Test Loss: {test_loss:.4f}, "
#                 f"Test accuracy: {accuracy:.4f}"
#             )
#       wandb.log({'Learning Rate': config.learning_rate, 'Batch Size': config.batch_size, 'Hidden Size': config.hidden_size, 'Test Loss': test_loss, 'Train Loss': train_loss, 'Model': model, 'Attention size': config.atten, 'Accuracy': accuracy, 'Epoch': epoch})
#     print("Time for training using PyTorch is %f" %(time() - start_time))
# wandb.agent(sweep_id, function=train)

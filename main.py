# Habitatable Exoplanets 
# CS 349 Machine Learning Final Project
# Authors: Charles Cheng and Hunter Cordes 
 
# General Imports 
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import xlsxwriter
# Torch Imports
from torch import tensor, float32, cuda, no_grad, cat, mean
from torch import abs as tAbs
from torch.nn import Module, Linear, Tanh, MSELoss
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
# SKLearn Imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

########## Classes ##########

class CustomDataset(Dataset):
  def __init__(self, data, labels, scaler):
    self.data = data
    self.labels = labels
    self.sc = scaler

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    raw = self.data[idx]
    rawl = self.labels[idx]
    if type(idx) == int:
      raw = raw.reshape(1, -1)
      rawl = [rawl]
      
    x = self.sc.transform(raw[:, 1:]) # exclude planet names
    data = tensor(x, dtype=float32)
    label = tensor(rawl, dtype=float32)
    return data, label

class FeedForwardNN(Module):
  def __init__(self):
    super(FeedForwardNN, self).__init__()
    self.linear1 = Linear(49, 32)
    self.relu1 = Tanh()
    self.linear2 = Linear(32, 16)
    self.relu2 = Tanh()
    self.linear_out = Linear(16, 1)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu1(x)
    x = self.linear2(x)
    x = self.relu2(x)
    x = self.linear_out(x)
    return x.squeeze().unsqueeze(1)
  
class PolynomialRegression(Module):
  def __init__(self, input_size, degree):
      super(PolynomialRegression, self).__init__()
      self.degree = degree
      self.fc = Linear(degree*input_size, 1)

  def forward(self, x):
      x_poly = cat([x.squeeze() ** i for i in range(1, self.degree + 1)], dim=1)
      return self.fc(x_poly)

class Datasets():
    def __init__(self, train_pname, train_data, train_labels, train_ld, valid_pname, valid_data, valid_labels, valid_ld, test_pname, test_data, test_labels, test_ld):
      self.train_pname = train_pname
      self.train_data = train_data
      self.train_labels = train_labels
      self.train_ld = train_ld
      self.valid_pname = valid_pname
      self. valid_data = valid_data
      self.valid_labels = valid_labels
      self.valid_ld = valid_ld
      self.test_pname = test_pname
      self.test_data = test_data
      self.test_labels = test_labels
      self.test_ld = test_ld

########## Helper functions ##########

def train(dataloader, model, loss_func, optimizer, device):
  model.train()
  train_loss = []

  now = datetime.datetime.now()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    pred = model(X)
    loss = loss_func(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 10 == 0:
      loss, current = loss.item(), batch * len(X)
      iters = 10 * len(X)
      then = datetime.datetime.now()
      try:
        iters /= (then - now).total_seconds()
        # print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
      except:
        iters /= 1
      now = then
      train_loss.append(loss)
  return train_loss

def test(dataloader, model, loss_func, device):
  size = len(dataloader)
  num_batches = 0
  model.eval()
  test_loss = 0

  with no_grad():
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_func(pred, y).item()
        num_batches = num_batches + 1
  test_loss /= num_batches
  # print(f"Avg Loss: {test_loss:>8f}\n")
  return test_loss

def plotLearnCurves(train_loss, valid_loss, name='default'):
  plt.figure()
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Training: Loss vs Epochs")
  plt.plot([i for i in range(len(train_loss))], tensor(train_loss).mean(axis=1))
  plt.savefig(name+"train.png")
  plt.close()

  plt.figure()
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Validation: Loss vs Epochs")
  plt.plot([i for i in range(len(valid_loss))], valid_loss)
  plt.savefig(name+"valid.png")
  plt.close()

def calcTestMSE(model, data):
  pred = model(data.test_data[:][0]).squeeze()
  tru = tensor(data.test_labels)
  mse_loss = F.mse_loss(pred, tru)
  print("Mean squared error =", mse_loss.item())

  abs_errors = tAbs(pred - tru)
  average_abs_error = mean(abs_errors).item()
  print("Average absolute error =", average_abs_error)

  return mse_loss.item(), average_abs_error

########## Main functions ##########

# Import and preprocess data 
def readData():
    df = pd.read_csv('phl_exoplanet_catalog.csv')
    # Features descriptions: https://phl.upr.edu/projects/habitable-exoplanets-catalog/hec-data-of-potentially-habitable-worlds/phls-exoplanets-catalog 

    # Remove samples with no ESI score 
    df = df.dropna(subset=['P_ESI'])

    #  Delete error columns 
    error_columns = [col for col in df.columns if 'ERROR' in col]
    df = df.drop(columns=error_columns)
    print("Number of error columns:", len(error_columns))

    # Delete columns where majority is unknown 
    num_col = len(df.columns)
    threshold = len(df) * 0.8 # hyper-parameter 
    df = df.dropna(axis=1, thresh=threshold)
    print("Number of empty columns:", num_col-len(df.columns))

    # Delete non-numeric columns
    # TODO: may want to encode non-numerical values? 
    num_col = len(df.columns)
    pname_column = df.pop("P_NAME")
    df = df.select_dtypes(include='number')
    df.insert(0, 'P_NAME', pname_column)
    print("Number of non-numeric columns:", num_col-len(df.columns))

    # Remove non-related features
    df = df.drop(columns=['P_YEAR', 'P_HABITABLE'])

    # Drop samples where majority of features is unknown 
    threshold = len(df.columns) * 0.8 # hyper-parameter 
    df = df.dropna(thresh=threshold)

    # P_ESI is the label 
    p_esi_column = df.pop("P_ESI")

    print("Number of features:", len(df.columns)-1) # Exclude the planet names
    print("Number of samples:", len(df))
    print("Features:", df.columns)

    # Replace missing values with the means of the columns
    pname_column = df.pop("P_NAME")
    df.fillna(df.mean(), inplace=True)
    df.insert(0, 'P_NAME', pname_column)

    # Statistics 
    df_data_stats = df.describe(include='number')
    print("\nFeature Statistics:")
    print(df_data_stats)
    df_labels_stats = p_esi_column.describe()
    print("\nESI Statistics:")
    print(df_labels_stats)

    # Excel sheet output
    writer = pd.ExcelWriter("features_statistics.xlsx", engine="xlsxwriter")
    df_data_stats.to_excel(writer, sheet_name="Sheet1", startrow=1, header=False, index=False)
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    (max_row, max_col) = df_data_stats.shape
    column_settings = [{"header": column} for column in df_data_stats.columns]
    worksheet.add_table(0, 0, max_row, max_col - 1, {"columns": column_settings})
    worksheet.set_column(0, max_col - 1, 12)
    writer.close()

    # Split samples in training, validation, and testing sets 
    data = df.values
    labels = p_esi_column.values

    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.3, random_state=17)
    valid_data, test_data, valid_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=17)
    print("\nTraining set size:", len(train_data))
    print("Validation set size:", len(valid_data))
    print("Testing set size:", len(test_data))

    sc = MinMaxScaler()
    sc.fit(train_data[:, 1:]) # exclude planet names 
    
    # Planet names 
    train_pname = train_data[:, 0]
    valid_pname = valid_data[:, 0]
    test_pname = test_data[:, 0]

    train_data = CustomDataset(train_data, train_labels, sc)
    valid_data = CustomDataset(valid_data, valid_labels, sc)
    test_data = CustomDataset(test_data, test_labels, sc)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    return Datasets(train_pname, train_data, train_labels, train_loader, valid_pname, valid_data, valid_labels, valid_loader, test_pname, test_data, test_labels, test_loader)
   
# Multivariate linear regression 
def polyReg(data, deg, lr=1e-4, max_epoch=10, loss_thres=1e-2, run=0):
    print("\n=============== Polynomial Regression ===============\n")

    device = "cuda" if cuda.is_available() else "cpu"
    input_size = data.train_data.data.shape[1] - 1  # Exclude the planet names
    poly_model = PolynomialRegression(input_size, deg).to(device)
    loss_func = MSELoss()
    optimizer = Adam(poly_model.parameters(), lr=lr)

    epochs = 0
    train_loss = []
    valid_loss = [] 
    while epochs < max_epoch and not (len(valid_loss) >= 2 and abs(valid_loss[-1] - valid_loss[-2]) < loss_thres):
        print(f"Epoch {epochs + 1}", end='\r')

        losses = train(data.train_ld, poly_model, loss_func, optimizer, device)
        train_loss.append(losses)
        valid_loss.append(test(data.valid_ld, poly_model, loss_func, device))
        epochs += 1

    plotLearnCurves(train_loss, valid_loss, name=f"poly_reg_{run}")
    sq_err, abs_err = calcTestMSE(poly_model, data)
    return sq_err, abs_err

# Feed forward neutral network 
def FFN(data, lr=1e-4, max_epoch=10, loss_thres=1e-2, run=0):
    print("\n=============== Feed Forward Neural Network ===============\n")

    device = "cuda" if cuda.is_available() else "cpu"
    ff = FeedForwardNN().to(device)
    loss_func = MSELoss() 
    optimizer = Adam(ff.parameters(), lr=lr)

    epochs = 0
    train_loss = []
    valid_loss = []
    while epochs < max_epoch and not (len(valid_loss) >= 2 and abs(valid_loss[-1]-valid_loss[-2]) < loss_thres): 
        print(f"Epoch {epochs+1}", end='\r')
        losses = train(data.train_ld, ff, loss_func, optimizer, device)
        train_loss.append(losses)
        valid_loss.append(test(data.valid_ld, ff, loss_func, device))
        epochs += 1
    
    plotLearnCurves(train_loss, valid_loss, name=f"ffn_{run}")
    return calcTestMSE(ff, data)

# Main function 
def habitExo():
    data = readData()
    # 1e-4/1e-6 200 epoc
    #FFN(data, lr=1e-4, max_epoch=50, loss_thres=1e-4)
    # 1e-3/1e-4 20 epoc 1e-4/1e-6 50 epoc 1e-4/1e-2 200 epoc
    # 1e-4/1e-5 30/130 epoc
    #polyReg(data, deg=1, lr=1e-2, max_epoch=200, loss_thres=1e-1, run="test")
    testing_poly(data)
  
def testing_poly(data):
   lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
   loss_threses = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
   #lr = [1e-3, 1e-4]
   #loss_threses = [1e-4, 1e-5]
   config_sq = [10, (0,0)]
   config_abs = [10, (0,0)]
   for l in lr:
    for L in loss_threses:
      sq_err, abs_err = FFN(data, lr=l, max_epoch=200, loss_thres=L, run=f"LR{l} LS{L}")
      if sq_err < config_sq[0]:
         config_sq[0] = sq_err
         config_sq[1] = (l,L)
      if abs_err < config_abs[0]:
         config_abs[0] = abs_err
         config_abs[1] = (l,L)
      print(f"Run: {l}/{L}")
   print(f"Configeration Square Means: {config_sq}")
   print(f"Configeration Absolute Means: {config_abs}")

if __name__ == "__main__":
    habitExo()
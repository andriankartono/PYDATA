import torch
import math
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


def function(x):
    return np.sin(x)


x_values = np.linspace(-math.pi, math.pi+0.1, 400)
y_values = function(x_values)

# Reshape to better suit the shape of a neural network
x_values = x_values.reshape((len(x_values), 1))
y_values = y_values.reshape((len(y_values), 1))


# Scale to improve robustness
scale_x = StandardScaler()
x_values = scale_x.fit_transform(x_values)
scale_y = StandardScaler()
y_values = scale_y.fit_transform(y_values)

x_values = torch.tensor(x_values)
y_values = torch.tensor(y_values)
train_data = (x_values, y_values)

loaders = {
    'train': torch.utils.data.DataLoader(
        train_data,
        batch_size=50,
        shuffle=True
    )
}


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        # NN Layer
        self.layer = nn.Sequential(
            nn.Linear(1, 200),
            nn.Tanh(),
            nn.Linear(200, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        output = self.layer(x)
        return output


# Define some global variables
model = NN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 700
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.L1Loss()
optimizer, model

model.train()
figure = plt.figure(figsize=(10, 8))
cols, rows = 3, 3
i = 1
x_plot, y_plot = scale_x.inverse_transform(
    x_values), scale_y.inverse_transform(y_values)

for epoch in range(num_epochs):
    for (x, y) in loaders["train"]:
        b_x = Variable(x)
        b_y = Variable(y)

        model_prediction = model(b_x.float())
        loss = loss_func(model_prediction, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 75 == 0:
            #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            figure.add_subplot(rows, cols, i)
            plt.plot(x_plot, y_plot)
            plt.plot(scale_x.inverse_transform(b_x.detach().numpy()),
                     scale_y.inverse_transform(model_prediction.detach().numpy()))
            plt.xticks(fontsize=4)
            plt.title("after {} epochs; MAE: {:.3f}".format(
                epoch+1, loss.item()))
            i += 1

plt.savefig("Aufgabe9_3.png")
plt.show()

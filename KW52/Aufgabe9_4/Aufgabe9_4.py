import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datas
train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

#
loaders = {
    'train': torch.utils.data.DataLoader(train_data,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=1),

    'test': torch.utils.data.DataLoader(test_data,
                                        batch_size=100,
                                        shuffle=True,
                                        num_workers=1),
}


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Second Convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization


def train(num_epochs, cnn, loaders):
    cnn.train()
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def test():
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    return accuracy


cnn = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)
num_epochs = 10


def main():
    train(num_epochs, cnn, loaders)
    test()
    # Create a list that stores all our false predictions
    false_predictions_test = []
    false_predictions_train = []

    for sample in iter(loaders["test"]):
        imgs, lbls = sample
        actual_number = lbls[:100].numpy()
        test_output, last_layer = cnn(imgs[:100])
        pred_y = torch.max(test_output, 1)[1].numpy()

        indexes, = np.where(pred_y != actual_number)
        for idx in indexes:
            false_predictions_test.append(
                (pred_y[idx], actual_number[idx], imgs[idx]))

    for sample in iter(loaders["train"]):
        imgs, lbls = sample
        actual_number = lbls[:600].numpy()
        test_output, last_layer = cnn(imgs[:600])
        pred_y = torch.max(test_output, 1)[1].numpy()

        indexes, = np.where(pred_y != actual_number)
        for idx in indexes:
            false_predictions_train.append(
                (pred_y[idx], actual_number[idx], imgs[idx]))

    test_accuracy = 1 - (len(false_predictions_test)/10000)
    train_accuracy = 1-(len(false_predictions_train)/60000)
    # Plot the false predictions
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    figure.suptitle("Train accuracy: {:.3f}, Test accuracy: {:.3f}".format(
        train_accuracy, test_accuracy))

    for i in range(1, cols * rows+1):
        plot_pred, plot_true, plot_image = false_predictions_test[i-1]
        figure.add_subplot(rows, cols, i)
        plt.imshow(plot_image.squeeze(), cmap="gray")
        plt.title(f"Pred:{plot_pred}, True:{plot_true}")
        plt.axis("OFF")

    plt.savefig("Aufgabe9_4.png")
    plt.show()


if __name__ == '__main__':

    main()

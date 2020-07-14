import model
import torch
import torch.nn as nn
from data import dir_xy, ImageLoader
import torch.optim as optim

train_hist=[]
test_hist=[]
def train(net=None, criterion=None, optimizer=None, TrainImgLoader=None, TestImgLoader=None, epochs=20):
    running_loss = 0.0
    test_loss = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(TrainImgLoader):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)  # 输出为[-1，2]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 ==9:
                print('第{}圈train Loss: {}'.format(epoch, running_loss / 10))
                train_hist.append(running_loss / 10)
                running_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(TestImgLoader):
                images, labels = data
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                if i % 10 ==9:
                    print('第{}圈test Loss: {}'.format(epoch, test_loss / 10))
                    test_hist.append(test_loss)
                    test_loss = 0.0
    plter(epochs=epochs, train_loss=train_hist, test_loss=test_hist)
    print('Finished Training')

def plter(train_loss,test_loss,epochs):
    import matplotlib.pyplot as plt
    x = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(range(len(train_loss)), train_loss, label='train')
    ax.plot(range(len(test_loss)), test_loss, label='test')
    ax.set_xlabel(xlabel='epoch')
    ax.set_ylabel(ylabel='MSE')
    ax.set_title('Epochs VS MSE')
    ax.legend()
    plt.show()

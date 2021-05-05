# (C) Copyright Artificial Brain 2021.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import torchvision
from torchvision import datasets
import torch
from quantumcat.applications.classification import Net
import torch.optim as optim
import torch.nn as nn
import time

class Classifier:
    "Loading the data"

    def __init__(self, path, provider):
        super(Classifier, self).__init__()
        # Concentrating on the first 100 samples
        self.path = path
        self.provider = provider
        n_samples = 100
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        X_train = datasets.MNIST(root = self.path, train = True, download = True,
                                 transform = transform)

        # Leaving only labels 0 and 1 
        idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                        np.where(X_train.targets == 1)[0][:n_samples])

        X_train.data = X_train.data[idx]
        X_train.targets = X_train.targets[idx]

        self.train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

        n_samples = 50

        X_test = datasets.MNIST(root = self.path, train = False, download = True, transform = transform)

        idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                        np.where(X_test.targets == 1)[0][:n_samples])

        X_test.data = X_test.data[idx]
        X_test.targets = X_test.targets[idx]

        self.test_loader = torch.utils.data.DataLoader(X_test, batch_size = 1, shuffle = True)

        #return train_loader

        self.model = Net()
        print(self.model)
        optimizer = optim.SGD(self.model.parameters(), lr = 0.001)
        self.loss_func = nn.NLLLoss()
        epochs = 20
        time0 = time.time()
        loss_list = []
        for epoch in range(epochs):
            self.total_loss = []
            target_list = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                target_list.append(target.item())
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                optimizer.step()
                self.total_loss.append(loss.item())
            loss_list.append(sum(self.total_loss)/len(self.total_loss))
            print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))

        # Normalise the loss between 0 and 1
        print("Training finished, took {:.2f}s  after epoch #{:2d}".format(time.time() - time0,epochs))

    def predict(self):
        self.model.eval()
        with torch.no_grad():

            correct = 0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                output = self.model(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss = self.loss_func(output, target)
                self.total_loss.append(loss.item())

        print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
            sum(self.total_loss) / len(self.total_loss),
            correct / len(self.test_loader) * 100)
            )
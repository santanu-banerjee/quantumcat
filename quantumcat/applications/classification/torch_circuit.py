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


import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from quantumcat.applications.classification import ClassifierCircuit
from quantumcat.utils import helper
import numpy as np


class TorchCircuit(Function):

    @staticmethod
    def forward(ctx, i):
        if not hasattr(ctx, 'Circuit'):
            ctx.Circuit = ClassifierCircuit(1)

        exp_value = ctx.Circuit.run(i[0])

        result = torch.tensor([exp_value]) # store the result as a torch tensor

        ctx.save_for_backward(result, i)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        s = np.pi/2

        forward_tensor, i = ctx.saved_tensors

        # Obtain paramaters
        input_numbers = helper.to_numbers(i[0])

        gradient = []

        for k in range(len(input_numbers)):
            input_plus_s = input_numbers
            input_plus_s[k] = input_numbers[k] + s  # Shift up by s

            exp_value_plus = ctx.Circuit.run(torch.tensor(input_plus_s))[0]
            result_plus_s = torch.tensor([exp_value_plus])

            input_minus_s = input_numbers
            input_minus_s[k] = input_numbers[k] - s # Shift down by s

            exp_value_minus = ctx.Circuit.run(torch.tensor(input_minus_s))[0]
            result_minus_s = torch.tensor([exp_value_minus])

            gradient_result = (result_plus_s - result_minus_s)

            gradient.append(gradient_result)

        result = torch.tensor([gradient])

        return result.float() * grad_output.float()


qc = TorchCircuit.apply


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.h1 = nn.Linear(320, 50)
        self.h2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.h1(x))
        x = F.dropout(x, training=self.training)
        x = self.h2(x)
        x = qc(x)
        x = (x+1)/2  # Normalise the inputs to 1 or 0
        x = torch.cat((x, 1-x), -1)
        return x
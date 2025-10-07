import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NetCNN(nn.Module):
    def __init__(self):
        super(NetCNN, self).__init__()
        # inception_1:
        self.first_conv_avg = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.first_norm_1 = nn.BatchNorm1d(16)
        self.first_pooling_avg = nn.AvgPool1d(3, stride=2)
        self.first_conv_max = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.first_norm_2 = nn.BatchNorm1d(16)
        self.first_pooling_max = nn.MaxPool1d(3, stride=2)
        self.first_conv = nn.Conv1d(16, 16, (60,), stride=(1,))
        self.first_norm_3 = nn.BatchNorm1d(16)

        # inception_2:
        self.second_conv_avg = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.second_norm_1 = nn.BatchNorm1d(16)
        self.second_pooling_avg = nn.AvgPool1d(3, stride=2)
        self.second_conv_max = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.second_norm_2 = nn.BatchNorm1d(16)
        self.second_pooling_max = nn.MaxPool1d(3, stride=2)
        self.second_conv = nn.Conv1d(16, 16, (50,), stride=(1,))
        self.second_norm_3 = nn.BatchNorm1d(16)

        # inception_3:
        self.third_conv_avg = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.third_norm_1 = nn.BatchNorm1d(16)
        self.third_pooling_avg = nn.AvgPool1d(3, stride=2)
        self.third_conv_max = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.third_norm_2 = nn.BatchNorm1d(16)
        self.third_pooling_max = nn.MaxPool1d(3, stride=2)
        self.third_conv = nn.Conv1d(16, 16, (40,), stride=(1,))
        self.third_norm_3 = nn.BatchNorm1d(16)

        # inception_4:
        self.fourth_conv_avg = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.fourth_norm_1 = nn.BatchNorm1d(16)
        self.fourth_pooling_avg = nn.AvgPool1d(3, stride=2)
        self.fourth_conv_max = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.fourth_norm_2 = nn.BatchNorm1d(16)
        self.fourth_pooling_max = nn.MaxPool1d(3, stride=2)
        self.fourth_conv = nn.Conv1d(16, 16, (30,), stride=(1,))
        self.fourth_norm_3 = nn.BatchNorm1d(16)

        # inception_5:
        self.fifth_conv_1 = nn.Conv1d(16, 16, (10,), stride=(1,))
        self.fifth_norm_1 = nn.BatchNorm1d(16)
        self.fifth_conv_2 = nn.Conv1d(16, 16, (20,), stride=(1,))
        self.fifth_norm_2 = nn.BatchNorm1d(16)

        # finally concat
        self.pooling2 = nn.MaxPool1d(3, stride=2)
        self.pooling3 = nn.AvgPool1d(3, stride=3)

        # fc
        self.fc1 = nn.Linear(16 * 13, 125)
        self.fc2 = nn.Linear(125, 75)
        self.fc3 = nn.Linear(75, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        x = X[:, :, :-1]
        # PART1
        # inception 1
        inc1 = F.leaky_relu_(self.first_norm_3(self.first_conv(x)))  # 16*1
        # 16*50  inception 2
        x = torch.concat([self.first_pooling_avg(F.leaky_relu_(self.first_norm_1(self.first_conv_avg(x)))),
                          self.first_pooling_max(F.leaky_relu_(self.first_norm_2(self.first_conv_max(x))))], dim=2)

        inc2 = F.leaky_relu_(self.second_norm_3(self.second_conv(x)), 1)  # 16*1
        # 16*40  inception 3
        x = torch.concat([self.second_pooling_avg(F.leaky_relu_(self.second_norm_1(self.second_conv_avg(x)))),
                          self.second_pooling_max(F.leaky_relu_(self.second_norm_2(self.second_conv_max(x))))],
                         dim=2)

        inc3 = F.leaky_relu_(self.third_norm_3(self.third_conv(x)), 1)  # 16*1
        # 16*30  inception 4
        x = torch.concat([self.third_pooling_avg(F.leaky_relu_(self.third_norm_1(self.third_conv_avg(x)))),
                          self.third_pooling_max(F.leaky_relu_(self.third_norm_2(self.third_conv_max(x))))], dim=2)

        inc4 = F.leaky_relu_(self.fourth_norm_3(self.fourth_conv(x)), 1)  # 16*1
        # 16*20  inception 5
        x = torch.concat([self.fourth_pooling_avg(F.leaky_relu_(self.fourth_norm_1(self.fourth_conv_avg(x)))),
                          self.fourth_pooling_max(F.leaky_relu_(self.fourth_norm_2(self.fourth_conv_max(x))))],
                         dim=2)

        inc5 = F.leaky_relu_(self.fifth_norm_2(self.fifth_conv_2(x)))  # 16*1
        inc6 = F.leaky_relu_(self.fifth_norm_1(self.fifth_conv_1(x)))  # 16*11

        # PART2
        inc = torch.concat([inc1, inc2, inc3, inc4, inc5, inc6], dim=2)  # 16*16
        inc = torch.concat([self.pooling3(inc), self.pooling2(inc)], dim=2)  # 16*12

        # PART3
        inc = torch.concat([X[:, :, -1].reshape([x.size(0), 16, 1]), inc], dim=2)  # 16*13
        inc = inc.view(x.size(0), -1)

        # PART4
        inc = self.fc1(inc)
        inc = F.leaky_relu_(self.fc2(inc))
        inc = self.fc3(inc)
        return inc


# train loop
def train_loop(data_loader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch, (x, y) in enumerate(data_loader):
        predictions = model(x)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# test loop
def test_loop(data_loader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            predictions = model(x)
            total_loss = loss_fn(predictions, y)
            test_loss += total_loss.item()
    return test_loss / len(data_loader)



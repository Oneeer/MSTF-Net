import os
import glob
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import MSTF
from loguru import logger
import numpy as np


# train_test_dataset
def train_test_data(files, lead_time):
    end_line = int(lead_time + 60)
    x = []
    y = []
    for i_file in range(0, len(files), 3):
        data = pd.concat([pd.read_csv(f) for f in files[i_file:i_file + 3]], axis=0, ignore_index=True)
        for i_line in range(0, (int(len(data)-end_line)), 30):
            if data.iloc[i_line, 0][0: 4] in ['2015', '2016']:
                if np.isnan(data.iloc[i_line:(i_line + 61), 1:]).any().sum() == 0:
                    if data.loc[i_line + end_line, 'Tsur'] <= 0 or data.loc[i_line + end_line, 'Tsur'] > 55:
                        x.append(data.iloc[i_line:(i_line + 61), 1:].T)
                        y.append(data.loc[i_line + end_line, 'Tsur'])
                    else:
                        pass
                else:
                    pass
            else:
                if np.isnan(data.iloc[i_line:(i_line + 61), 1:]).any().sum() == 0 and ~np.isnan(
                        data.loc[i_line + end_line, 'Tsur']):
                    x.append(data.iloc[i_line:(i_line + 61), 1:].T)
                    y.append(data.loc[i_line + end_line, 'Tsur'])
                else:
                    pass
    x = torch.tensor(np.array(x), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    y = (y + 15) / (80 + 15)
    return x, y


if __name__ == '__main__':
    # Model initialize
    logger.info('model setting---------------------------')
    model = MSTF.NetCNN()
    learning_rate = 0.001
    epoch = 500
    batch_size = 64
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss()

    # Open original dataset
    logger.info('----------------train_test_dataset---------------')
    files = glob.glob(os.path.join('Model_stations/*.csv'))

    for i_time in range(60, 730, 60):
        # Generating train_test_dataset with i_time lead time
        logger.info('MSTF with ' + str(int(i_time/60)) + 'hour')
        x, y = train_test_data(files, i_time)
        dataset = TensorDataset(x, y)
        torch.manual_seed(0)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train, test = random_split(dataset, lengths=[train_size, test_size])
        train_data_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

        # Train with i_time lead time
        train_loss = []
        test_loss = []
        for i_epoch in range(epoch):
            logger.info('train epoch' + str(i_epoch+1))
            train_loss.append(MSTF.train_loop(train_data_loader, model, criterion, optimizer))
            test_loss.append(MSTF.test_loop(test_data_loader, model, criterion))
            # 保存模型
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i_epoch + 1
            }
            pathname = 'F:/TNN/MSTFWithDifferentLeadTime/Checkpoint_' + str(int(i_time/60)) + 'hour'
            if not os.path.isdir(pathname):
                os.mkdir(pathname)
            torch.save(state, pathname + '/epoch_%d.ckpt' % (i_epoch + 1))

            total_loss = pd.concat([pd.DataFrame(train_loss), pd.DataFrame(test_loss)], axis=1)
            total_loss.columns = ['Train_loss', 'Test_loss']
            total_loss.to_csv(pathname + '/LearningCurve_' + str(int(i_time/60))+ 'hour.csv', index=True)










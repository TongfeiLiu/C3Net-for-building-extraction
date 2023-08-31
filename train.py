from model.CPSCNet import *

from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torchvision.transforms as Transforms
import torch.utils.data as data
import time
import torch
import glob
import os
from utils.EvaluationNew import Evaluation
from utils.EvaluationNew import Index
from utils.loss import dice_loss
import cv2
import numpy as np
from tqdm import tqdm

#
def train_net(net, device, data_path, val_path, ModelName='CPSCNet_out1_BCE_out2_Dice', epochs=100, batch_size=2, lr=0.0001):
    # print(net)
    # Load dataset
    isbi_dataset = ISBI_Loader(data_path, transform=Transforms.ToTensor())
    train_loader = data.DataLoader(dataset=isbi_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], gamma=0.9)

    # criterion = nn.BCEWithLogitsLoss()
    BCE_loss = nn.BCELoss()
    f_loss = open('train_loss.txt', 'w')
    f_time = open('train_time.txt', 'w')
    # Start training epochs
    for epoch in range(epochs):
        net.train()

        num = int(0)
        starttime = time.time()

        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch), colour='white') as t:
            for image, label in train_loader:
                optimizer.zero_grad()

                image = image.to(device=device)
                label = label.to(device=device)

                pred1, pred, out = net(image)

                # compute loss
                bce_Loss = BCE_loss(pred1, label)
                dice_Loss = dice_loss(pred, label)
                loss = 0.5 * bce_Loss + 0.5 * dice_Loss
                # dice_Loss = dice_loss(out, label)
                # loss = criterion(pred2, label)

                if num == 0:
                    if epoch == 0:
                        f_loss.write('Note: epoch (num, edge_loss, focal_loss, BCE_loss, total_loss)\n')
                        f_loss.write('epoch = ' + str(epoch) + '\n')
                    else:
                        f_loss.write('epoch = ' + str(epoch) + '\n')
                f_loss.write(str(num) + ',' + str(float('%5f' % loss)) + '\n')

                loss.backward()
                optimizer.step()
                num += 1
                t.set_postfix({'lr': '%.5f' % optimizer.param_groups[0]['lr'],
                                'loss': '%.4f' % (loss.item()),})
                t.update(1)
        # learning rate delay
        scheduler.step()

        endtime = time.time()
        if epoch == 0:
            f_time.write('each epoch time\n')
        f_time.write(str(epoch)+','+str(starttime)+','+str(endtime)+','+str(float('%4f' % (endtime-starttime))) + '\n')
        # val
        if epoch > 4:
            with torch.no_grad():
                mOA, IoU = val(net, device, epoch, val_path)
                # best_F1 = F1
                # print(str(epoch) + ':::::OA=' + str(float('%2f' % (mOA))) + ':::::mIoU=' + str(float('%2f' % (IoU))))

                modelpath = str(ModelName) + '_BestmIoU_' + 'epoch_' + str(epoch) + '_mIoU_' + str(
                    float('%2f' % IoU)) + '.pth'

                torch.save(net.state_dict(), modelpath)
    f_loss.close()
    f_time.close()

def val(net, device, epoc, val_DataPath):
    net.eval()
    image = glob.glob(os.path.join(val_DataPath, 'image/*.tif'))
    label = glob.glob(os.path.join(val_DataPath, 'label/*.tif'))
    trans = Transforms.Compose([Transforms.ToTensor()])
    IoU, c_IoU, uc_IoU, OA, Precision, Recall, F1 = 0, 0, 0, 0, 0, 0, 0
    num = 0
    TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or = 0, 0, 0, 0, 0, 0
    val_acc = open('val_acc.txt', 'a')
    val_acc.write('===============================' + 'epoch=' + str(epoc) + '==============================\n')
    with tqdm(total=len(image), desc='Val Epoch #{}'.format(epoc), colour='yellow') as t:
        for val_path, label_path in zip(image, label):
            num += 1

            val_img = cv2.imread(val_path)
            val_label = cv2.imread(label_path)
            val_label = cv2.cvtColor(val_label, cv2.COLOR_BGR2GRAY)
            val_img = trans(val_img)

            val_img = val_img.unsqueeze(0)

            val_img = val_img.to(device=device, dtype=torch.float32)

            pred1, pred, out = net(val_img)

            # acquire result
            pred = np.array(pred.data.cpu()[0])[0]
            # binary map
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            monfusion_matrix = Evaluation(label=val_label, pred=pred)
            TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
            TPSum += TP
            TNSum += TN
            FPSum += FP
            FNSum += FN
            C_Sum_or += c_num_or
            UC_Sum_or += uc_num_or

            if num > 30:
                Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
                IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
                OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()

            t.set_postfix({
                           'OA': '%.4f' % OA,
                           'mIoU': '%.4f' % IoU,
                           'c_IoU': '%.4f' % c_IoU,
                           'uc_IoU': '%.4f' % uc_IoU,
                           'PRE': '%.4f' % Precision,
                           'REC': '%.4f' % Recall,
                           'F1': '%.4f' % F1})
            t.update(1)
    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    val_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                  str(float('%2f' % (c_IoU))) + ',' +
                  'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                  'PRE = ' + str(float('%2f' % (Precision))) + ',' +
                  'REC = ' + str(float('%2f' % (Recall))) + ',' +
                  'F1 = ' + str(float('%2f' % (F1))) + '\n')
    val_acc.close()
    return OA, IoU

if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    net = CPSCNet(n_channels=3, n_classes=1)
    net.to(device=device)

    # load building extraction datasets
    ###########################################
    #-(Austin/Chicago/Kitsap/Tyrol/Vienna)-#
    # data_path = "./data/Inria/Vienna/train"
    # val_path = "./data/Inria/Vienna/test"
    ###########################################
    data_path = "./data/EastAsia/train"
    val_path = "./data/EastAsia/test"
    ###########################################
    # data_path = "./data/Massachusetts512/train"
    # val_path = "./data/Massachusetts512/test"
    ###########################################

    train_net(net, device, data_path, val_path)

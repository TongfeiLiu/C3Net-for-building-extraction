import glob
import numpy as np
import cv2

from model.CPSCNet import *
from utils.EvaluationNew import Evaluation
from utils.EvaluationNew import Index
import torchvision.transforms as Transforms
import time
from tqdm import tqdm

if __name__ == "__main__":
    print('Starting test...')
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    net = CPSCNet(n_channels=3, n_classes=1)
    net.to(device=device)

    # load model
    net.load_state_dict(torch.load('CPSCNet.pth', map_location=device))
    net.eval()

    trans = Transforms.Compose([Transforms.ToTensor()])
    # Inria-(Austin/Chicago/Kitsap/Tyrol/Vienna)-# EastAsia/Massachusetts512
    dataset = 'EastAsia'
    MethodName='C3Net'
    img_path = glob.glob('./data/' + str(dataset) +'/test/image/*.tif')
    label_path = glob.glob('./data/' + str(dataset) + '/test/label/*.tif')
    # img_path = glob.glob('./data/Inria/' + str(dataset) +'/test/image/*.tif')
    # label_path = glob.glob('./data/Inria/' + str(dataset) + '/test/label/*.tif')
    # 遍历所有图片
    num = 1
    IoU, c_IoU, uc_IoU, OA, Precision, Recall,   F1 = 0, 0, 0, 0, 0, 0, 0
    TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or = 0, 0, 0, 0, 0, 0
    f_acc = open(str(dataset) + '_test_acc.txt', 'w')
    f_time = open(str(dataset) + '_test_time.txt', 'w')
    with tqdm(total=len(img_path), desc='Test Epoch #{}'.format(num), ncols=160, colour='cyan') as t:
        for tests_path, label_path in zip(img_path, label_path):
            starttime = time.time()
            # 保存结果地址
            save_res_path = tests_path.split('.')[1] + '_res' +  '.png'
            save_res_path = '.' + save_res_path.replace('image', 'results')
            name = tests_path.split('/')[5].split('.')[0]
            # 读取图片
            test_img = cv2.imread(tests_path)
            label_img = cv2.imread(label_path)
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
            test_img = trans(test_img)
            test_img = test_img.unsqueeze(0)
            test_img = test_img.to(device=device, dtype=torch.float32)

            pred1, pred_Img, out = net(test_img)

            pred = np.array(pred_Img.data.cpu()[0])[0]

            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0

            cv2.imwrite(save_res_path, pred)
            endtime = time.time()

            if num == 0:
                f_time.write('each pair images time\n')
            f_time.write(str(num) + ',' + str(starttime) + ',' + str(endtime) + ',' + str(
                float('%4f' % (endtime - starttime))) + '\n')

            # evaluate
            monfusion_matrix = Evaluation(label=label_img, pred=pred)
            TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
            TPSum += TP
            TNSum += TN
            FPSum += FP
            FNSum += FN
            C_Sum_or += c_num_or
            UC_Sum_or += uc_num_or
            # save loss和accuracy
            if num == 1:
                f_acc.write('=================================================================================\n')
                f_acc.write('|Note: (num, FileName, TP, TN, FP, FN)|\n')
                f_acc.write('|Note: (ACC: FileName, OA, FA, MA, TE, mIoU, c_IoU, uc_IoU, Precision, Recall, F1)|\n')
                f_acc.write('=================================================================================\n')

            f_acc.write(str(num) + ',' + str(name) + '.tif' + ',' + str(TP) + ',' + str(TN) + ',' +
                        str(FP) + ',' + str(FN) + '\n')

            num += 1
            if num > 30:
                Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
                IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
                OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
                FA, MA, TE = Indicators.CD_indicators()
            t.set_postfix({
                           'OA': OA,
                           'mIoU': '%.4f' % IoU,
                           'c_IoU': '%.4f' % c_IoU,
                           'uc_IoU': '%.4f' % uc_IoU,
                           'PRE': '%.4f' % Precision,
                           'REC': '%.4f' % Recall,
                           'F1': '%.4f' % F1})
            t.update(1)

    Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
    OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    FA, MA, TE = Indicators.CD_indicators()

    f_acc.write('==========================================================================================================\n')
    f_acc.write('|SumConfusionMatrix:|  TP   |   TN   |  FP  |  FN   |\n')
    f_acc.write('|SumConfusionMatrix:|' + str(TPSum) + '|' + str(TNSum) + '|' + str(FPSum) + '|' + str(FNSum) + '|\n')
    f_acc.write('==========================================================================================================\n')
    f_acc.write('|TotalAcc:|   OA   |   FA   |   MA    |  TE   |  mIoU   |  c_IoU  | uc_IoU  |Precision| Recall  |   F1    |\n')
    f_acc.write('|TotalAcc:|' + str(float('%4f' % OA)) + '|' + str(float('%4f' % FA)) + '|' + str(float('%4f' % MA)) + '|' + str(float('%4f' % TE))
                + '|' + str(float('%4f' % IoU)) + '|' + str(float('%4f' % c_IoU)) + '|' + str(float('%4f' % uc_IoU)) + '|' +
                str(float('%4f' % Precision)) + '|' + str(float('%4f' % Recall)) + '|' + str(float('%4f' % F1)) + '|\n')
    f_acc.write(
        '==========================================================================================================\n')

    f_acc.close()
    f_time.close()
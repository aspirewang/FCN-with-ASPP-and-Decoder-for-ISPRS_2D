import sklearn.metrics as metric
from PIL import Image
import numpy as np
import os

image_list = os.listdir('network_result_1D')
OA=np.array([0])
F1=np.array([0 ,0 ,0 ,0 ,0])
PRECISION=np.array([0 ,0 ,0 ,0 ,0])
RECALL=np.array([0 ,0 ,0 ,0 ,0])

for name in image_list:
    print(name)
    prediction = Image.open('network_result_1D/' + name)
    prediction = np.array(prediction)
    prediction = np.reshape(prediction, (5632*5632))

    label = Image.open('5_Labels_all_1D/' + name)
    label = np.array(label)
    label = label[0:5632, 0:5632]
    label = np.reshape(label, (5632*5632))


    oa = metric.accuracy_score(label, prediction)
    print(oa)
    OA = OA + oa

    f1 = metric.f1_score(label, prediction, average=None)
    f1 = f1[0:5]
    F1 = F1 + f1

    precision = metric.precision_score(label, prediction, average=None)
    precision = precision[0:5]
    PRECISION = PRECISION + precision

    recall = metric.recall_score(label, prediction, average=None)
    recall = recall[0:5]
    RECALL = RECALL + recall

OA = OA / 14
F1 = F1 / 14
PRECISION = PRECISION/14
RECALL = RECALL/14
OA_average = np.average(OA)
F1_average = np.average(F1)
PRECISION_average = np.average(PRECISION)
RECALL_average = np.average(RECALL)

print('OA:', OA_average, OA)
print('F1:', F1_average, F1)
print('precision:', PRECISION_average, PRECISION)
print('recall:', RECALL_average, RECALL)
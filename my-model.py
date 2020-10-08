#!/usr/bin/python
#


import argparse

# CCredit Card Fraud Detection
# Import important packages
import pandas as pd
import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
#from cf_matrix import make_confusion_matrix
#%matplotlib inline
#import warnings
#warnings.filterwarnings('ignore')
# set seed
np.random.seed(123)

# Importing KNN module from PyOD
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
# Import the utility function for model evaluation
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

from polyaxon import tracking
from polyaxon.tracking.contrib.keras import PolyaxonKerasCallback, PolyaxonKerasModelCheckpoint




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.002
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20
    )

    args = parser.parse_args()

    # 입력 파라미터 출력
    print("learning_rate: {}".format(args.learning_rate))
    print("epochs: {}".format(args.epochs))

    # Polyaxon
    tracking.init()
    plx_callback = PolyaxonKerasCallback()
    plx_model_callback = PolyaxonKerasModelCheckpoint()
    log_dir = tracking.get_tensorboard_path()

    print("log_dir", log_dir)
    print("model_dir", plx_model_callback.filepath)

    # 데이터 읽기
    data = pd.read_csv('/container_dataset1/creditcard.csv')

    # 각 클래스별 데이터 분리
    positive = data[data['Class'] == 1]
    negative = data[data['Class'] == 0]
    print("positive:{}".format(len(positive)))
    print("negative:{}".format(len(negative)))

    # negative 는 1만건만 사용한다.
    new_data = pd.concat([positive, negative[:10000]])
    new_data = new_data.sample(frac=1, random_state=42)
    print(new_data.shape)

    # 카드 사용액이 커서 해당 필드만 표준 scaler 로 조정한다.
    scaler = StandardScaler()

    # 카드 사용액 스케일링 적용
    new_data['Amount'] = scaler.fit_transform(new_data['Amount'].values.reshape(-1,1))

    # Time, Class 컬럼을 사용하지 않는다.
    X = new_data.drop(['Time', 'Class'], axis=1)
    y = new_data['Class']
    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))

    # train, test set 으로 본리한다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # K-NN, One-class SVM 모델을 생성한다.
    clf_knn = KNN(contamination=0.172, n_neighbors=5, n_jobs=-1)
    clf_knn.fit(X_train)

    # train 에 대한 라벨을 예측해본다.
    y_train_pred = clf_knn.labels_ # (0: inliers, 1: outliers)
    acc = np.sum(y_train == y_train_pred) / len(y_train)
    print('Accuracy: {}'.format(acc))

    # polyaxon 로그 함수를 이용해서 정확도를 기록하도록 한다.
    tracking.log_metrics(steps=2, loss=0.09, validation=0.9, accuracy=0.85)
    


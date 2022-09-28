import pandas as pd
import torch
import random
from imblearn.over_sampling import SMOTE
from numpy import savetxt



def data():
    data = pd.read_csv("creditcard.csv")
    data = data.dropna()
    # balance data
    grouped = data.groupby('Class')
    # extract labels
    y = torch.tensor(data["Class"].values).float().unsqueeze(1)
    data = data.drop(columns = "Class")
    # standardize data
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    return split_train_test(x, y)



def split_train_test(x, y, test_ratio=0.3):

    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    # delimiter between test and train data
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]

    sm = SMOTE(sampling_strategy = 0.0068, n_jobs = -1)
    x_train_res, y_train_res = sm.fit_resample(x[train_idxs], y[train_idxs])
    x_test_res, y_test_res = sm.fit_resample(x[test_idxs], y[test_idxs])

    return x_train_res, y_train_res, x_test_res, y_test_res     # tipo numpy.ndarray



x_train, y_train, x_test, y_test = data()

# proporção original da ocorrência da classe minoritária 0.0017
# na de treinamento (len de 199365) e na de teste (len de 85442).

# proporção após modificação da ocorrência da classe minoritária 0.0068
# na de treinamento (len de 200362) e na de teste (len de 85886).

savetxt('x_train_res.csv',  x_train, delimiter=',')
savetxt('y_train_res.csv',  y_train, delimiter=',')
savetxt('x_test_res.csv',  x_test, delimiter=',')
savetxt('y_test_res.csv',  y_test, delimiter=',')


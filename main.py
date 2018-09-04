import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn import svm
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
SCORE = make_scorer(f1_score)
KERNEL = ['rbf']
GAMMA = np.logspace(-3, 3, 7, base=10)
COST = np.logspace(-3, 3, 7, base=10)


def plot_data(df, target):
    split_data = [df[df.Survived == survived] for survived in [0, 1]]
    temp = [data[target].dropna() for data in split_data]
    plt.hist(temp, histtype="barstacked", bins=16)
    plt.show()


def main():
    print('\n-------------------------------------------------------\n')
    print('kernal: ', KERNEL)
    print('Gamma: ', GAMMA)
    print('COST: ', COST)
    print('\n-------------------------------------------------------\n')

    df_train = pd.read_csv(TRAIN_DATA_PATH).replace(['male', 'female'], [0, 1]).replace(['C', 'S', 'Q'], [0, 1, 2])
    df_train['Age'].fillna(df_train.Age.mean(), inplace=True)
    df_train['Embarked'].fillna(df_train.Embarked.mean(), inplace=True)
    x_train = df_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    y_train = df_train['Survived']
    x_train = MinMaxScaler().fit_transform(x_train)

    svc = svm.SVC()
    parameter = {'kernel': KERNEL, 'C': COST, 'gamma': GAMMA}
    clf = GridSearchCV(svc, parameter, scoring='accuracy', n_jobs=-1, cv=5, verbose=3, return_train_score=False)
    clf.fit(x_train, y_train)

    print('\n-------------------------------------------------------\n')
    print('Best Estimator:\n', clf.best_estimator_)
    print('Best Score:', clf.best_score_)
    print('\n-------------------------------------------------------\n')

    df_test = pd.read_csv(TEST_DATA_PATH).replace(['male', 'female'], [0, 1]).replace(['C', 'S', 'Q'], [0, 1, 2])
    x_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    x_test['Age'].fillna(x_test.Age.median(), inplace=True)
    x_test['Fare'].fillna(x_test.Fare.mean(), inplace=True)
    x_test = MinMaxScaler().fit_transform(x_test)

    test_pred = clf.predict(X=x_test)
    df_pred = pd.DataFrame(test_pred, columns=['Survived'])
    df_test['Survived'] = df_pred['Survived']
    df_test[['PassengerId', 'Survived']].to_csv('submit/{0:%Y%m%d%H%M}.csv'.format(datetime.now()), index=False)
    print('End Program')


if __name__ == '__main__':
    main()
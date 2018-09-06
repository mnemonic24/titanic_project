import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
KERNEL = ['rbf', 'linear']
GAMMA = np.logspace(-3, 3, 7, base=10)
COST = np.logspace(-3, 3, 7, base=10)
NAME_LIST = ['Mr', 'Mrs', 'Miss', 'Master']
TRAIN_DROP_LIST = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
TEST_DROP_LIST = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
CV = 10


def plot_data(df, target):
    split_data = [df[df.Survived == survived] for survived in [0, 1]]
    temp = [data[target].dropna() for data in split_data]
    plt.hist(temp, histtype="barstacked", bins=16)
    plt.show()


def name_mean(df):
    for name in NAME_LIST:
        mean = df['Age'][df.Name.str.contains(name)].dropna().mean()
        df.loc[df['Name'].str.contains(name), 'Age'] = df.loc[df['Name'].str.contains(name), 'Age'].fillna(mean)
    return df


def processing_data(path):
    df = pd.read_csv(path)
    print(df.columns)
    print(df.info())
    print('NaN numbers:')
    print(df.isnull().sum())
    print('\n-------------------------------------------------------\n')

    df = name_mean(df)
    df.fillna(df.mean(), inplace=True)
    df = pd.concat([df, pd.get_dummies(df[['Sex', 'Embarked']])], axis=1)
    print(df.isnull().sum())
    print('\n-------------------------------------------------------\n')
    return df


def main():
    print('\n-------------------------------------------------------\n')
    print('kernal: ', KERNEL)
    print('Gamma: ', GAMMA)
    print('COST: ', COST)
    print('\n-------------------------------------------------------\n')

    df_train = processing_data(TRAIN_DATA_PATH)
    x_train = df_train.drop(TRAIN_DROP_LIST, axis=1)
    y_train = df_train['Survived']
    x_train = MinMaxScaler().fit_transform(x_train)
    print('\n-------------------------------------------------------\n')

    df_test = processing_data(TEST_DATA_PATH)
    x_test = df_test.drop(TEST_DROP_LIST, axis=1)
    x_test = MinMaxScaler().fit_transform(x_test)
    print('\n-------------------------------------------------------\n')

    svc = svm.SVC()
    parameter = {'kernel': KERNEL, 'C': COST, 'gamma': GAMMA}
    clf = GridSearchCV(svc, parameter, scoring='accuracy', n_jobs=-1, cv=CV, verbose=3, return_train_score=False)
    clf.fit(x_train, y_train)

    print('Best Estimator:\n', clf.best_estimator_)
    print('Best Score:', clf.best_score_)
    print('\n-------------------------------------------------------\n')

    test_pred = clf.predict(X=x_test)
    df_pred = pd.DataFrame(test_pred, columns=['Survived'])
    df_test['Survived'] = df_pred['Survived']
    df_test[['PassengerId', 'Survived']].to_csv('submit/{0:%Y%m%d%H%M}.csv'.format(datetime.now()), index=False)
    print('End Program')


if __name__ == '__main__':
    main()

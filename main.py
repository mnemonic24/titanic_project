import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Activation
import xgboost as xgb

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
KERNEL = ['rbf', 'linear']
GAMMA = np.logspace(-3, 3, 7, base=10)
COST = np.logspace(-3, 3, 7, base=10)
NAME_LIST = ['Mr', 'Mrs', 'Miss', 'Master']
TRAIN_DROP_LIST = ['PassengerId', 'Survived', 'Ticket', 'Cabin']
TEST_DROP_LIST = ['PassengerId', 'Ticket', 'Cabin']
CV = 10
optimizer = ['adam', 'sgd']
nb_epoch = [10]
batch_size = [16]


def plot_data(df, target):
    split_data = [df[df.Survived == survived] for survived in [0, 1]]
    temp = [data[target].dropna() for data in split_data]
    plt.hist(temp, histtype="barstacked", bins=16)
    plt.show()


def name_mean(train, test, xMat):
    for name in NAME_LIST:
        mean = xMat['Age'][xMat.Name.str.contains(name)].dropna().mean()
        train.loc[train['Name'].str.contains(name), 'Age'].fillna(mean, inplace=True)
        test.loc[test['Name'].str.contains(name), 'Age'].fillna(mean, inplace=True)

    train.drop(['Name'], axis=1, inplace=True)
    test.drop(['Name'], axis=1, inplace=True)
    return train, test


def processing_data(path):
    df = pd.read_csv(path).replace(['male', 'female'], [0, 1]).replace(['C', 'S', 'Q'], [0, 1, 2])
    return df


def scaling(df, mean):
    df.fillna(mean, inplace=True)
    df = pd.get_dummies(df)
    df = MinMaxScaler().fit_transform(df)
    return df


def clf_svc():
    svc = SVC()
    parameter = {'kernel': KERNEL, 'C': COST, 'gamma': GAMMA}
    clf = GridSearchCV(svc, parameter, scoring='accuracy', n_jobs=-1, cv=CV, verbose=3, return_train_score=False)
    return clf


def clf_xgb():
    model = xgb.XGBClassifier()
    parameter = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}
    clf = GridSearchCV(model, parameter, scoring='accuracy', n_jobs=-1, cv=CV, verbose=3, return_train_score=False)
    return clf


def build_model():
    model = Sequential()
    model.add(Dense(16, input_dim=7, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def clf_nn():
    model = KerasClassifier(build_fn=build_model)
    parameter = dict(nb_epoch=nb_epoch, batch_size=batch_size)
    clf = GridSearchCV(model, parameter, n_jobs=-1, cv=CV, verbose=1, return_train_score=False)
    return clf


def clf_rf():
    model = RandomForestClassifier(n_jobs=-1)
    parameter = {'max_depth': [10], 'min_samples_split': [5, 10],
                 'min_samples_leaf': [1, 5], 'n_estimators': [n for n in range(10, 100, 10)]}
    clf = GridSearchCV(model, parameter, n_jobs=-1, cv=CV, verbose=3, return_train_score=False)
    return clf


def main():
    df_train = processing_data(TRAIN_DATA_PATH)
    df_test = processing_data(TEST_DATA_PATH)

    X = df_train.drop(TRAIN_DROP_LIST, axis=1)
    y = df_train['Survived']
    x_test = df_test.drop(TEST_DROP_LIST, axis=1)

    Xmat = pd.concat([X, x_test])
    X, x_test = name_mean(X, x_test, Xmat)

    X = scaling(X, Xmat.mean())
    x_test = scaling(x_test, Xmat.mean())

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    print('x train shape', x_train.shape)
    print('y train shape', y_train.shape)

    print('\n--------------------------------------------------------\n')

    clf = clf_xgb()
    clf.fit(x_train, y_train)

    print('Best Estimator:\n', clf.best_estimator_)
    print('Best Score:', clf.best_score_)

    print('\n--------------------------------------------------------\n')

    y_pred = clf.predict(x_valid)
    score = clf.score(x_valid, y_valid)
    print('Validation Score: ', score)
    print('Classification report: ')
    print(classification_report(y_valid, y_pred))

    test_pred = clf.predict(x_test)
    df_pred = pd.DataFrame(test_pred, columns=['Survived'])
    df_test['Survived'] = df_pred['Survived']
    df_test[['PassengerId', 'Survived']].to_csv('submit/{0:%Y%m%d%H%M}.csv'.format(datetime.now()), index=False)
    print('End Program')


if __name__ == '__main__':
    main()

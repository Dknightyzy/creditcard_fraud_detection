# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import tools
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from itertools import cycle
from sklearn.pipeline import Pipeline
from sklearn .grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

if __name__ == '__main__':
    start = time.clock()
    # 划分数据集
    data = tools.load('creaditcard.pkl')
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    X = data.ix[:, data.columns != 'Class']
    y = data.ix[:, data.columns == 'Class']

    number_records_fraud = len(data[data.Class == 1])
    fraud_indices = np.array(data[data.Class == 1].index)
    normal_indices = np.array(data[data.Class == 0].index)

    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)

    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    under_sample_data = data.iloc[under_sample_indices, :]

    X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
    y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,
                                                                                                        y_undersample,
                                                                                                        test_size=0.3,
                                                                                                        random_state=0)
    '''----------------------------------------------------------------------'''
    # # logistic regression
    # pipe_logistic = Pipeline([('clf', LogisticRegression())])
    # param_range = [0.01, 0.1, 1, 10, 100]
    # param_grid = [{'clf__C': param_range, 'clf__penalty': ('l1', 'l2')}]
    # gs = GridSearchCV(estimator=pipe_logistic, param_grid=param_grid, scoring='recall', cv=10, n_jobs=1)
    # gs = gs.fit(X_train_undersample, y_train_undersample.values.ravel())
    # print gs.grid_scores_
    # print gs.best_params_
    #
    # clf = gs.best_estimator_
    # print 'Test accuracy: %.3f' % clf.score(X.values, y.values.ravel())
    # y_pred_proba = clf.predict_proba(X)
    # # 作图
    # tools.recall_proba(y, y_pred_proba)
    # plt.show()

    '''--------------------------------------------------------------------'''
    # # svm
    # pipe_svm = Pipeline([('clf', SVC(random_state=1))])
    # param_range = [0.01, 0.1, 1, 10, 100]
    # param_grid = [  # {'clf__C': param_range, 'clf__kernel': ['linear']},
    #               {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
    # gs = GridSearchCV(estimator=pipe_svm, param_grid=param_grid, scoring='roc_auc', cv=10, n_jobs=-1)
    # gs = gs.fit(X_train_undersample, y_train_undersample.values.ravel())
    # print gs.grid_scores_
    # print gs.best_params_
    #
    # clf = gs.best_estimator_
    # clf.fit(X_train_undersample, y_train_undersample.values.ravel())
    # # print 'Test accuracy: %.3f' % clf.score(X.values, y.values.ravel())
    # y_pred = clf.predict(X)
    # tools.plot_cm(y, y_pred)
    # plt.show()

    '''-----------------------------------------------------------------------'''
    # RandomForest
    forest = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=1, n_jobs=-1)
    forest.fit(X_train_undersample, y_train_undersample)
    y_pred = forest.predict(X)

    pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=1))])
    param_grid = [{'clf__criterion': ['entropy', 'gini'], 'clf__n_estimators': [10, 100, 200]}]
    gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='roc_auc', cv=10, n_jobs=-1)
    gs = gs.fit(X_train_undersample, y_train_undersample.values.ravel())
    print gs.grid_scores_
    print gs.best_params_

    clf = gs.best_estimator_
    clf.fit(X_train_undersample, y_train_undersample.values.ravel())
    # print 'Test accuracy: %.3f' % clf.score(X.values, y.values.ravel())
    y_pred = clf.predict(X)
    tools.plot_cm(y, y_pred)
    plt.show()

    '''----------------------------------------------------------------------'''
    # # ensemble learning
    # clf1 = LogisticRegression(penalty='l1', C=0.01)
    # clf2 = SVC(C=1.0, kernel='rbf', gamma=0.01)
    # clf3 = RandomForestClassifier(criterion='entropy', n_estimators=200)
    # mv_clf = VotingClassifier(estimators=[('LR', clf1),
    #                                       ('SVM', clf2),
    #                                       ('MV', clf3)], voting='soft')
    # clf_labels = ['Logistic Regression', 'SVM', 'RandomForest', 'MajorityVoting']
    # all_clf = [clf1, clf2, clf3, mv_clf]
    # for clf, label in zip(all_clf, clf_labels):
    #     scores = cross_val_score(estimator=clf, X=X_train_undersample, y=y_train_undersample.values.ravel(), cv=5, scoring='roc_auc', n_jobs=-1)
    #     print "Accuracy: %f (+/- %f) [%s]" % (scores.mean(), scores.std(), label)

    end = time.clock()
    print "耗时：%f s" %(end - start)
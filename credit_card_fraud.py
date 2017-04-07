# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tools
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from itertools import cycle

# data = pd.read_csv("creditcard.csv")
# data.head()
# pickle_.dump(data)

data = tools.load('creaditcard.pkl')
# print data.head()

# 画图查看各类标签数量统计
# count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
# count_classes.plot(kind='bar')
# plt.title("Fraud class histogram")
# plt.xlabel('Class')
# plt.ylabel('Frequency')
# plt.show()

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
# print data.head()

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

# # Showing ratio
# print"Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 1])/float(len(under_sample_data))
# print"Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/float(len(under_sample_data))
# print("Total number of transactions in resampled data: ", len(under_sample_data))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,
                                                                                                    y_undersample,
                                                                                                    test_size=0.3,
                                                                                                    random_state=0)

best_c = tools.printing_Kfold_scores(X_train_undersample, y_train_undersample, visual=False)

# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C=best_c, penalty='l2')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)

print"在欠采样测试集上的召回率:", float(cnf_matrix[1, 1])/(cnf_matrix[1, 0]+cnf_matrix[1, 1])

# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
tools.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
# plt.show()


# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C=best_c, penalty='l2')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

print"在全体测试集上的召回率:", float(cnf_matrix[1,1])/(cnf_matrix[1, 0]+cnf_matrix[1, 1])

# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
tools.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
# plt.show()


# ROC CURVE
lr = LogisticRegression(C=best_c, penalty='l2')
y_pred_undersample_score = lr.fit(X_train_undersample, y_train_undersample.values.ravel()).decision_function(X_test_undersample.values)

fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(), y_pred_undersample_score)
roc_auc = auc(fpr, tpr)
# print roc_auc

# Plot ROC
plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()


lr = LogisticRegression(C=0.01, penalty='l2')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(10, 10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

    plt.subplot(3, 3, j)
    j += 1

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print"Recall metric in the testing dataset: ", float(cnf_matrix[1, 1]) / (cnf_matrix[1, 0] + cnf_matrix[1, 1])

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    tools.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' % i)

# plt.show()


lr = LogisticRegression(C=0.01, penalty='l2')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue', 'black'])

plt.figure(figsize=(5, 5))

j = 1
for i, color in zip(thresholds, colors):
    y_test_predictions_prob = y_pred_undersample_proba[:, 1] > i

    precision, recall, thresholds = precision_recall_curve(y_test_undersample, y_test_predictions_prob)

    # Plot Precision-Recall curve
    plt.plot(recall, precision, color=color,
             label='Threshold: %s' % i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example')
    plt.legend(loc="lower left")

# plt.show()


# 在全集上预测
lr = LogisticRegression(C=0.01, penalty='l2')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_proba = lr.predict_proba(X.values)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(10, 10))

j = 1
for i in thresholds:
    y_test_predictions = y_pred_proba[:, 1] > i

    plt.subplot(3, 3, j)
    j += 1

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y, y_test_predictions)
    np.set_printoptions(precision=2)

    print"Recall metric in the testing dataset: ", float(cnf_matrix[1, 1]) / (cnf_matrix[1, 0] + cnf_matrix[1, 1])

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    tools.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' % i)


# prc图
lr = LogisticRegression(C=0.01, penalty='l2')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X.values)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue', 'black'])

plt.figure(figsize=(5, 5))

j = 1
for i, color in zip(thresholds, colors):
    y_test_predictions_prob = y_pred_undersample_proba[:, 1] > i

    precision, recall, thresholds = precision_recall_curve(y, y_test_predictions_prob)

    # Plot Precision-Recall curve
    plt.plot(recall, precision, color=color,
             label='Threshold: %s' % i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example')
    plt.legend(loc="lower left")
plt.show()
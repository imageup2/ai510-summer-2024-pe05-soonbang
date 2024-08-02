from sklearn import svm

train_data = [[10], [100]]
train_target = [1, 0]
clf = svm.SVC()
clf.fit(train_data, train_target)

import joblib
joblib.dump(clf, "binary_clf.joblib")

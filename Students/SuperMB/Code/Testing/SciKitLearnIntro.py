from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

print(iris.data)
print(digits.data)

print(digits.target)

from sklearn import svm
print("Create support vector classifier")
clf = svm.SVC(gamma=0.001, C=100.)
print("Fit data")
clf.fit(digits.data[:-1], digits.target[:-1])
print("Predict!")
print(clf.predict(digits.data[-1:]))



print("Test saving a model")
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0:1]))
print(y[0])

print("Dump to file")
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl')
print("Load")
clf = joblib.load('filename.pkl')

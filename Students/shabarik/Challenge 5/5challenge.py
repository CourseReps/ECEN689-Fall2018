import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

student_identity = 'shabarik'

filenameToRead = '5challenge_training_' + student_identity + '.csv'
training_df = pd.read_csv(filenameToRead)
testing_df = pd.read_csv('5challenge_testing_shabarik.csv')

print(training_df.head())

X1 = training_df[['Feature 0', 'Feature 1']].values
X2 = testing_df[['Feature 0', 'Feature 1']].values
Y1 = training_df[['Class']].values
Y1 = Y1.ravel()
#plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1.ravel(), s=25, edgecolor='k')
#plt.grid()
#plt.xlabel('Feature 0')
#plt.ylabel('Feature 1')

svm1 = SVC(kernel='linear')
svm2 = SVC(kernel='poly')
svm3 = SVC(kernel='sigmoid')
svm4 = SVC(kernel='rbf')
svm1.fit(X1,Y1)
svm2.fit(X1,Y1)
svm3.fit(X1,Y1)
svm4.fit(X1,Y1)
#print(svm.score(X1,Y1))
Y11 = svm1.predict(X2)
Y12 = svm2.predict(X2)
Y13 = svm3.predict(X2)
Y14 = svm4.predict(X2)

# c_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# parameters = dict(gamma=gamma_range, C=c_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# OptGrid = GridSearchCV(SVC(), param_grid=parameters, cv=None)
# OptGrid.fit(X1, Y1)

#print(OptGrid.best_params_, OptGrid.best_score_)

svm5 = SVC(C=100.0, kernel='rbf', gamma=1.0)
svm5.fit(X1,Y1)
Y15 = svm5.predict(X2)
testing_df['Class'] = Y15

plot_decision_regions(X=X1,y=Y1,clf=svm1)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Linear Kernel SVM')
plt.show()

plot_decision_regions(X=X1,y=Y1,clf=svm2)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Polynomial Kernel SVM')
plt.show()

plot_decision_regions(X=X1,y=Y1,clf=svm3)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Sigmoid Kernel SVM')
plt.show()

plot_decision_regions(X=X1,y=Y1,clf=svm5)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('RBF Kernel SVM')

testing_df.to_csv('5challenge_testing_shabarik_output.csv', index=False)

plt.show()

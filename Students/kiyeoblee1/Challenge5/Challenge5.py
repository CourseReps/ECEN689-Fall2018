import os
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

student_identity = 'kiyeoblee'

## Load up the data
os.chdir('/Users/kiyeoblee/Desktop/Course work/ECEN689-Fall2018/Challenges/')
filenameToRead = '5challenge_training_' + student_identity + '.csv'
training_df = pd.read_csv('5Files/' + filenameToRead)
filenameToRead2 = '5challenge_testing_' + student_identity + '.csv'
testing_df = pd.read_csv('5Files/' + filenameToRead2)

## Processing up the data
X1 = training_df[['Feature 0', 'Feature 1']].values
Y1 = training_df[['Class']].values
X2 = testing_df[['Feature 0', 'Feature 1']].values
Y2 = testing_df[['Class']].values
X_training = X1.tolist()
Y_training = Y1.tolist()
X_testing = X2.tolist()
Y_testing = Y2.tolist()
Y = [item for sublist in Y_training for item in sublist]
tr_class1 = training_df.iloc[training_df.index[(training_df.iloc[:,1] == 1) == True],2:]
tr_class2 = training_df.iloc[training_df.index[(training_df.iloc[:,1] == 0) == True],2:]

## Training and Prediction
clf = svm.SVC(kernel='rbf', gamma=20)
clf.fit(X_training, Y)
Y_testing = clf.predict(X_testing)
Y_testing = pd.DataFrame(Y_testing)
te_class1 = testing_df.iloc[Y_testing.index[(Y_testing.iloc[:,0] == 1) == True],2:]
te_class2 = testing_df.iloc[Y_testing.index[(Y_testing.iloc[:,0] == 0) == True],2:]
fileNameToWrite = '/Users/kiyeoblee/Desktop/prediction.csv'
Y_testing.to_csv(fileNameToWrite)

## Convex hull for plotting
hull1 = ConvexHull(tr_class2)
hull2 = ConvexHull(te_class2)
boundary = [[0.05, 1.93],[-0.93, -0.19],[0.57, -2.3],[0.98, -2.3],[0.64, 1.67]]
boundary = pd.DataFrame(boundary)
hull3 = ConvexHull(boundary)

## Plotting graphs
# colors = ['b','y']
# a1 = tr_class1.iloc[:,0];b1 = tr_class1.iloc[:,1]
# z1 = np.polyfit(a1.values.flatten(), b1.values.flatten(), 1)
# p1 = np.poly1d(z1)
# a2 = tr_class2.iloc[:,0];b2 = tr_class2.iloc[:,1]
# z2 = np.polyfit(a2.values.flatten(), b2.values.flatten(), 1)
# p2 = np.poly1d(z2)
# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# pl1 = ax1.scatter(tr_class1.iloc[:,0], tr_class1.iloc[:,1], marker='o', s=25, edgecolor='k', color=colors[0])
# pl2 = ax1.scatter(tr_class2.iloc[:,0], tr_class2.iloc[:,1], marker='o', s=25, edgecolor='k', color=colors[1])
# ax1.set_xlim((-4,4))
# ax1.set_ylim((-4,4))
# ax1.set_title('Training Data')
# plt.plot(a1,p1(a1),"b--",a2,p2(a2),"y--")
# plt.legend((pl1, pl2), ('Class 1', 'Class 2'),scatterpoints=1,loc='lower left',ncol=3,fontsize=8)

# a1 = tr_class1.iloc[:,0];b1 = tr_class1.iloc[:,1]
# z1 = np.polyfit(a1.values.flatten(), b1.values.flatten(), 1)
# p1 = np.poly1d(z1)
# a2 = tr_class2.iloc[:,0];b2 = tr_class2.iloc[:,1]
# z2 = np.polyfit(a2.values.flatten(), b2.values.flatten(), 1)
# p2 = np.poly1d(z2)
# ax2 = fig.add_subplot(2,1,2)
# pl3 = ax2.scatter(tr_class1.iloc[:,0], tr_class1.iloc[:,1], marker='o', s=25, edgecolor='k', color=colors[0])
# pl4 = ax2.scatter(tr_class2.iloc[:,0], tr_class2.iloc[:,1], marker='o', s=25, edgecolor='k', color=colors[1])
# ax2.set_xlim((-4,4))
# ax2.set_ylim((-4,4))
# ax2.set_title('Training Data')
# plt.plot(a1,p1(a1),"b--",a2,p2(a2),"y--")
# plt.legend((pl3, pl4), ('Class 1', 'Class 2'),scatterpoints=1,loc='lower left',ncol=3,fontsize=8)

# for simplex in hull1.simplices:
#     plt.plot(tr_class2.iloc[simplex, 0], tr_class2.iloc[simplex, 1], '-', color='black')
# for simplex in hull3.simplices:
#     plt.plot(boundary.iloc[simplex, 0], boundary.iloc[simplex, 1], '-', color='red')
# plt.show()


# ax2 = fig.add_subplot(2,1,2)
# pl3 = ax2.scatter(te_class1.iloc[:,0], te_class1.iloc[:,1], marker='o', s=25, edgecolor='k', color=colors[0])
# pl4 = ax2.scatter(te_class2.iloc[:,0], te_class2.iloc[:,1], marker='o', s=25, edgecolor='k', color=colors[1])
# ax2.set_xlim((-4,4))
# ax2.set_ylim((-4,4))
# ax2.set_title('Testing Data(Prediction)')
# plt.plot(a1,p1(a1),"b--",a2,p2(a2),"y--")
# plt.legend((pl3, pl4), ('Class 1', 'Class 2'),scatterpoints=1,loc='lower left',ncol=3,fontsize=8)
# for simplex in hull1.simplices:
#     plt.plot(tr_class2.iloc[simplex, 0], tr_class2.iloc[simplex, 1], '-', color='black')
# for simplex in hull3.simplices:
#     plt.plot(boundary.iloc[simplex, 0], boundary.iloc[simplex, 1], '-', color='red')
# plt.show()

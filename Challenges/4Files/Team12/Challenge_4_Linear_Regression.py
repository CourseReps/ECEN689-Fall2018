import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

class Linear_wine:

    def __init__(self, alpha = 0.001, tolerance = 0.1, max_iters = 100000):
        self.thetas =[]
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.alpha = alpha

    def train(self, X, Y):

        X = np.array(X,dtype=float)
        Y = np.array(Y,dtype=float)

        batch_start = 0
        batch_end = 100

        no_examples, no_features = np.shape(X)
        self.thetas = np.ones(no_features)


        X_T = np.array(X,dtype=float).T
        for i in range(self.max_iters):

            diffs = np.dot(X, self.thetas) - Y
            cost = np.sum(np.square(diffs)) / (2*no_examples)
            gradient = np.dot(X_T,diffs) / (no_examples)
            self.thetas = self.thetas - (self.alpha * gradient)
            print(cost)

            if cost < self.tolerance:
                break

    def set_thetas(self, thetas):

        self.thetas=thetas

    def predict(self, X):

        return np.dot(X, self.thetas)

    def score(self, X, Y):

        Y_pred = self.predict(X)
        rmse = sqrt(mean_squared_error(Y,Y_pred))
        print(self.thetas)
        print(rmse)




train = pd.read_csv('winequality-white-training.csv')
test = pd.read_csv('winequality-white-testing_with_labels.csv').dropna(axis=0)

train_y = train['quality'].values.tolist()
train_data = train.drop(['Id','quality'], axis=1)
#train_data['pH*alcohol'] = np.array(train_data['pH'].values.tolist(),dtype=float) * np.array(train_data['alcohol'].values.tolist(),dtype=float)
#train_data['sulfur'] = np.array(train_data['density'].values.tolist(),dtype=float) * np.array(train_data['total sulfur dioxide'].values.tolist(),dtype=float)
#test['pH*alcohol'] = np.array(test['pH'].values.tolist(),dtype=float) * np.array(test['alcohol'].values.tolist(),dtype=float)
train_values = train_data.values.tolist()
test_labels = test['quality'].values.tolist()
test_values = test.drop(['Id', 'quality'],axis=1).values.tolist()
test_without_labels_df = pd.read_csv('winequality-white-testing.csv')
test_without_labels = test_without_labels_df.drop('Id', axis=1).values.tolist()

# scale = StandardScaler(with_std=False)
# train_values = scale.fit_transform(train_values)
# test_values = scale.fit_transform(test_values)
# test_without_labels = scale.fit_transform(test_without_labels)

# no_rows, no_cols = np.shape(train_values)
#
# for i in range(0,no_cols):
#     data = []
#     data.append(np.array(train_data, dtype=float)[:,i].T)
#     data.append(np.array(train_y, dtype=float).T)
#     cov_mat = np.cov(data)
#     corr = cov_mat[0][1] / (sqrt(cov_mat[0][0] * sqrt(cov_mat[1][1])))
#     print('Feature ', i,': ',corr)

lin = Linear_wine()
#best_thetas = [7.50822055e-03, -1.02878326e+00, -7.59322751e-02, 4.54893404e-03, -1.77254862e+00, 6.55225479e-03, -3.88689141e-03, 4.27360548e+00, -4.87595095e-01, 9.89945560e-01, 2.93634289e-01]
best_thetas = [-3.83017762e-02, -1.96256042e+00, -6.30684124e-02,  2.95896188e-02, 2.90379994e-01, 6.04713634e-03, -1.08059584e-03, 1.11961522e+00, 3.79866241e-01, 3.31181660e-01, 3.76873879e-01]
lin.set_thetas(best_thetas)
#lin.train(train_values,train_y)
lin.score(test_values, test_labels)

predictions = lin.predict(test_without_labels)
full_array = []

for i in range(len(predictions)):

    full_array.append([i, predictions[i]])

labels = ['Id', 'quality']

wine_quality_red = pd.DataFrame(full_array, columns=labels)
wine_quality_red.to_csv('winequality-white-solution.csv', index=False)

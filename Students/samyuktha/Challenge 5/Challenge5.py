import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize

def features_class(train_dataframe):

    features = train_dataframe.iloc[:, 1:].values
    label = train_dataframe['Class'].values

    return features, label


def split_trainingData(features, label, train_percentage):
    """
    Split the dataset with train_percentage
    """

    # Split dataset into train and test dataset
    X_train, X_test, Y_train, Y_test = train_test_split(features, label, train_size=train_percentage)
    return X_train, X_test, Y_train, Y_test


def predictors_test(test_dataframe):
    """
    Get the predictors in testing dataset
    """

    Predictors = test_dataframe.iloc[:, 1:].values

    return Predictors

def supportVectorMachine(features, label, kernel):
    C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=45)
    grid = GridSearchCV(SVC(kernel=kernel), param_grid=param_grid, cv=cv)
    grid.fit(features, label)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    plotHeatmap(grid, C_range, gamma_range)

    return grid

def plotHeatmap(grid, C_range, gamma_range):
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=Normalize(vmin=0.2, vmax=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.savefig("Heat Map.png")
    plt.show()

def plotModel(x,y, grid, title):
    h = 0.01
    markers = ('o', 'x')
    colors = ('green', 'purple')


    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))

    Z = grid.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples
    for i, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1],
                    alpha=0.8, c=cmap(i),
                    marker=markers[i])
        plt.title("Plot using " + title + " kernel")
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.savefig(title + ".png")


def main():
    """
    Main Function
    :return:
    """

    #Load csv files onto dataframes

    train_dataframe = pd.read_csv("5challenge_training_ssankaran30.csv", index_col=0)
    test_dataframe  = pd.read_csv("5challenge_testing_ssankaran30.csv", index_col=0)

    #Split dataframes for features and class

    features, label = features_class(train_dataframe)

    # Split Training data into train and test datasets
    X_train, X_test, Y_train, Y_test = split_trainingData(features, label, 0.2)

    #Train model
    #For rbf kernel
    model = supportVectorMachine(features, label,kernel='rbf')



    #Prediciting on validation set
    prediction = model.predict(X_test)

    # Calculating Accuracy on Validation data set
    res = prediction - Y_test
    count = 0;
    length = len(prediction)
    for i in range(length):
        if(res[i] == 0):
            count += 1;
    accuracy = count/length * 100;
    print("accuracy = ",accuracy,"%");

    #Plot model
    plotModel(features, label, model,title='rbf')

    # Prediction on Test Data
    features, label = features_class(train_dataframe)
    testPredictors = predictors_test(test_dataframe)

    model.fit(features, label)
    Y_pred_test = model.predict(testPredictors)

    # saving predictions to the test file
    test_dataframe["Class"] = Y_pred_test
    test_dataframe.to_csv("5challenge_testing_ssankaran30.csv")

if __name__ == "__main__":
    main()











import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


class PopulationPredict:

    def __init__(self):

        self.train_data = pd.DataFrame()
        self.sparse_params = []

    def five_Sparse(self,X, Y, ind):

        print(ind)
        check = 1
        alp = 15
        itr = 1
        init = 4
        while (True):
            las = Lasso(alpha= alp, normalize=True)
            las.fit(X,Y)
            coefs = las.coef_.tolist()
            count = 0

            for coef in coefs:
                if coef != 0.0:
                    count +=1
            print('iteration: ',ind,'\ncount: ',count)
            if count>5:
                #init = 2
                alp+= init
                check=0
                itr+=1
            else:
                alp = 15
                check=1
                itr = 1
                init=4

            if itr>50:
                itr = 1
                init *= 2


            if(check):
                break

            #print(las.coef_)
        coefs.insert((ind),0.0)
        self.sparse_params.append(coefs)


    def train(self,X):

        self.train_data = X
        for ind, rows in X.iterrows():

            y = X.loc[ind, :]
            y = y.values.tolist()
            del y[0]
            x = X.copy()
            x = x.drop(index=ind)
            x = x.drop('Country Name', axis = 1).values.tolist()
            x = np.array(x).T
            #print(np.shape(y))
            self.five_Sparse(x,y, ind)
        labels = X['Country Name'].values.tolist()
        coef_mat = self.sparse_params.copy()
        coef_mat.insert(0,labels.copy())
        labels.insert(0,'Country Name')
        params_df = pd.DataFrame.from_records(coef_mat)
        params_df.to_csv('params.csv',index=False)

    def predict(self,Y = pd.DataFrame):

        my_df = Y.copy()
        my_df = my_df.drop['Country Name']
        df_list = Y.values.tolist()
        labels = Y.dtypes.index
        full_mat = []
        no_rows, no_cols = np.shape(df_list)
        for i in range(0,no_rows):
            curr_list = []
            for j in range(0,no_cols):
                this_year = np.array(df_list[:,j]).T
                val = self.sparse_params[i] * this_year
                curr_list.append(val)
            full_mat.append(curr_list)
        countries = Y['Country Name'].values.tolist()
        full_mat.insert(0, countries)
        output_df = pd.DataFrame.from_records(full_mat, columns=labels)
        output_df = output_df.transpose
        output_df.to_csv('output.csv',index=False)

    def prediction_from_params(self, Y=pd.DataFrame):

        params_df = pd.read_csv('params.csv')
        params_df.drop(params_df.index[:1], inplace=True)
        params = params_df.values.tolist()
        #print(np.shape(params))
        #print(params_df.head(3))

        self.sparse_params = params
        sparse_array = np.array(self.sparse_params,dtype=float)

        my_df = Y.copy()
        my_df = my_df.drop(['Country Name'], axis=1)
        df_list = np.array(my_df.values.tolist())
        labels = list(Y.dtypes.index)
        full_mat = []
        no_rows, no_cols = np.shape(df_list)
        for i in range(0,no_rows):
            curr_list = []
            for j in range(0,no_cols):
                this_year = np.array(df_list[:,j],dtype=float).T
                #val = sparse_array[i] * this_year
                val = np.dot(sparse_array[i],this_year)
                curr_list.append(val)
            full_mat.append(curr_list)
        countries = Y['Country Name'].values.tolist()
        del labels[0]
        full_mat.insert(0, labels)
        full_mat = np.array(full_mat).T.tolist()
        countries.insert(0,'Id')
        # for i in range(len(countries)):
        #     full_mat[i].insert(0,countries[i])
        output_df = pd.DataFrame.from_records(full_mat,columns=countries)
        #output_df.set_index('Country Name', inplace=True)
        output_df.to_csv('output.csv',index=False)




population_training_df = pd.read_csv('population_training.csv', encoding='cp1252').dropna(axis=0)
population_testing_df = pd.read_csv('population_testing.csv', encoding='cp1252').dropna(axis=0)
population_testing_df = population_testing_df.drop(index=123)
population_training_df = population_training_df.reset_index(drop=True)
population_testing_df = population_testing_df.reset_index(drop=True)

pop = PopulationPredict()
#pop.train(population_training_df)
#pop.predict(population_testing_df)
pop.prediction_from_params(population_testing_df)

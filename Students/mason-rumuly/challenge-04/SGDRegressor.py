import numpy as np
import matplotlib.pyplot as plt

# Custom SGD linear regressor for the fun of it

class SGDRegressor(object):
    def __init__(self, learning_rate, rate_decay=1, epochs=1, use_momentum=True, moment_factor=0.9):
        '''
        Sets learning_rate to learning_rate*rate_decay once per epoch
        Trains using L2 loss
        '''
        self.learning_rate = learning_rate
        self.rate_decay = rate_decay
        self.W = None
        self.b = None
        self.epochs = epochs
        self.use_momentum = use_momentum
        self.momentum = None
        self.moment_factor = moment_factor

    def initialize_coefs(self, length, method='zeros'):
        if method == 'zeros':
            self.W = np.zeros(length)
            self.b = 0
            if self.use_momentum:
                self.momentum = np.zeros(length)
        elif method == 'random':
            self.W = np.random.randn(length)
            self.b = np.random.randn()
            if self.use_momentum:
                self.momentum = np.random.randn(length)
        
        return self
    
    def fit(self, X, y, learning_rate=None, rate_decay=None, epochs=None, keep_hist=False):
        '''
        Fits to the given training data, discarding previous experience
        X should be 1D or 2D iterable
        Y should be 1D iterable
        '''
        # Prepare training data
        X_array = np.array(X).reshape((len(y), -1))
        
        # Initialize weight vector
        self.initialize_coefs(X_array.shape[1])

        # partial fit
        return self.partial_fit(X_array, y, learning_rate, rate_decay, epochs, keep_hist)

    def partial_fit(self, X, y, learning_rate=None, rate_decay=None, epochs=None, keep_hist=False):
        '''
        Moves toward given set
        '''
        # get the previously-defined values
        if learning_rate is None:
            learning_rate = self.learning_rate
        if rate_decay is None:
            rate_decay = self.rate_decay
        if epochs is None:
            epochs = self.epochs

        # Prepare training data
        X_array = np.array(X).reshape((len(y), -1))

        # Keep update history
        if keep_hist:
            self.W_hist = [np.zeros(self.W.shape) for _ in range(epochs+1)]
            self.b_hist = [0 for _ in range(epochs+1)]
            self.W_hist[0] = self.W
            self.b_hist[0] = self.b

        # for each epoch
        for e in range(epochs):
            # on each training/value pair
            for x_row, y_val in zip(X_array, y):

                # compute loss gradient
                difference = self.predict(x_row) - y_val
                lgrad = 2 * x_row * difference

                # create random vector with expectation of loss gradient
                rgrad = lgrad*(1 + np.random.randn(*self.W.shape))

                # values for update
                update = rgrad*learning_rate

                if self.use_momentum:
                    self.momentum = self.moment_factor*self.momentum - update
                    self.W += self.momentum
                else:
                    self.W -= update

                # update W
                self.W -= rgrad*learning_rate

                # update intercept
                self.b -= learning_rate*2*difference*(1+np.random.randn())
            
            # update history
            if keep_hist:
                self.W_hist[e] = self.W
                self.b_hist[e] = self.b

            # decay learning rate
            learning_rate *= rate_decay

        return self
    
    def predict(self, X):
        assert self.W is not None, 'not fit'

        X_array = np.array(X)

        # deal with single samples
        if len(X_array.shape) == 1 and (X_array.shape[0] == 1 or not self.W.shape[0] == 1):
            return self.W.dot(X_array) + self.b
        
        # deal with multiple samples
        X_array = X_array.reshape((-1, self.W.shape[0]))
        return (np.sum(self.W * X_array, axis=1) + self.b).reshape((-1,))

    def mse(self, X, y):
        '''
        Get mean squared error over given set
        '''
        difference = self.predict(X) - np.array(y)
        return (difference.dot(difference)).mean()
        

# Unit tests
if __name__ == '__main__':

    sgdr = SGDRegressor(0.01, epochs=100)
    # should work on 1D and 2D training data
    sgdr.fit(
        [1, 2, 3], [1, 2, 3]
    )
    assert sgdr.W.shape[0] == 1, 'Invalid coefficient size on 1D'
    print(sgdr.mse([1, 2, 3], [1, 2, 3]))
    sgdr.fit(
        [[1,1],[2,4],[3,9]], [2, 6, 12], learning_rate=0.001, epochs=1000, keep_hist=True
    )
    assert sgdr.W.shape[0] == 2, 'Invalid coefficient size on 2D'
    print(sgdr.mse([[1,1],[2,4],[3,9]], [2, 6, 12]))
    print(sgdr.W, sgdr.b) 
    

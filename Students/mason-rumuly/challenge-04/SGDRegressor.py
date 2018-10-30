import numpy as np

# Custom SGD linear regressor for the fun of it

class SGDRegressor(object):
    def __init__(
        self, 
        learning_rate=0.001, 
        rate_decay=1, 
        epochs=100, 
        use_momentum=True, 
        moment_factor=0.9,
        regularization='none',
        reg_strength = 0.1
    ):
        '''
        Sets learning_rate to learning_rate*rate_decay once per epoch
        Trains using L2 loss
        regularization values: 'none', 'lasso', 'ridge'
        '''
        self.learning_rate = learning_rate
        self.rate_decay = rate_decay
        self.W = None
        self.b = None
        self.epochs = epochs
        self.use_momentum = use_momentum
        self.momentum = None
        self.moment_factor = moment_factor
        self.regularization = regularization
        self.reg_strength = reg_strength

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
        y_array = np.array(y)

        # initializat if not already
        if self.W is None:
            self.initialize_coefs(X_array.shape[1])

        # Keep update history
        if keep_hist:
            self.W_hist = [np.zeros(self.W.shape) for _ in range(epochs+1)]
            self.b_hist = [0 for _ in range(epochs+1)]
            self.W_hist[0] = self.W
            self.b_hist[0] = self.b

        # for each epoch
        for e in range(epochs):
            # on each training/value pair
            for x_row, y_val in zip(X_array, y_array):

                # compute loss gradient
                difference = self.predict(x_row) - y_val
                lgrad = 2 * x_row * difference
                if self.regularization == 'lasso':
                    lgrad += self.reg_strength*np.sign(self.W)
                elif self.regularization == 'ridge':
                    lgrad += self.reg_strength*2*self.W

                # create random vector with expectation of loss gradient
                # sample is a random vector, don't need this stupidity
                rgrad = lgrad  # *(1 + np.random.randn(*self.W.shape))

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
                b_grad = learning_rate*2*difference
                if self.regularization == 'lasso':
                    b_grad += self.reg_strength*np.sign(self.b)
                elif self.regularization == 'ridge':
                    b_grad += self.reg_strength*2*self.b
                self.b -= b_grad*(1+np.random.randn())
            
            # update history
            if keep_hist:
                self.W_hist[e] = self.W
                self.b_hist[e] = self.b

            # decay learning rate
            learning_rate *= rate_decay

        return self
    
    def predict(self, X):
        '''
        predict values corresponding to X
        '''
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
        return ((self.predict(X) - np.array(y))**2).mean()
        
    def score(self, X, y):
        '''
        Get performance as (1-u/v), where u is sum of squared errors and v is the label variance. 
        Best score is 1, worst scores approach -infty
        '''
        y_array = np.array(y)
        return 1 - ((self.predict(X) - y_array)**2).sum()/((y_array - y_array.mean())**2).sum()

# Unit tests
if __name__ == '__main__':

    # simple test vectors, can be perfectly reproduced
    part_0 = [1, 2, 3], [1, 2, 3]
    part_1 = [[1,1],[2,4],[3,9]], [2, 6, 12]

    sgdr = SGDRegressor(0.01, epochs=100, regularization='lasso', reg_strength=0.001)
    # should work on 1D and 2D training data
    sgdr.fit(
        *part_0
    )
    assert sgdr.W.shape[0] == 1, 'Invalid coefficient size on 1D'
    print('MSE:', sgdr.mse(*part_0), 'Score:', sgdr.score(*part_0))
    sgdr.fit(
        *part_1, learning_rate=0.001, epochs=1000, keep_hist=True
    )
    assert sgdr.W.shape[0] == 2, 'Invalid coefficient size on 2D'
    print('MSE:', sgdr.mse(*part_1), 'Score:', sgdr.score(*part_1))
    

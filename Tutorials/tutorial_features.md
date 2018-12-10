# Introduction to Feature Engineering
by Anirudh Shaktawat and Kanchan Satpute

Nowadays, Everybody thinks that the better the features they will choose and prepare, the better the results they will achieve for their machine learning model! This is a true statement but at the same time it is a misleading statement. In reality, the results in a machine learning model depend on various factors (as shown in the image).

![Pic1](https://github.com/anirudh2312/deep-learning/blob/master/images/Picture1.png)

Then the question is "Why Feature Engineering" !The importance of Feature Engineering is evident in the three statements which are:

#### Better feature means better flexibility:
Better features lead to less complex models, Faster to run, Easier to understand and Easier to maintain.

#### Better feature means simpler models:
Even if you have chosen wrong parameters (not the most optimized ones) for your machine learning model, you may still get good results if you have better features. You do not need to work as hard to pick the right models and the most optimized parameters. 

#### Better feature means better results:
In fact, Xavier Conort (current grandmaster and former 1st ranker on Kaggle) once said, 'The algorithms we used are very standard for Kagglers. We spent most of our efforts in Feature Engineering', when asked in an interview when he won one of the toughest known challenges on Kaggle.

From the above arguments, it is clear that Feature Engineering is indeed important for success in machine learning, most of the times. Now, the question arises that "What is Feature Engineering"! Different people interpret this term in different ways:
1. Some think that extracting out features from raw data by reducing the dimensionality of observations into a much smaller set, is feature engineering. 
2. Others think that the process of scoring and ranking the features and selecting the best subset of features, is feature engineering
3. Most of the people think that manually creating new features using the given raw data, is feature engineering


### Feature Engineering:
Feature Engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on resulting data. 
![Pic2](https://github.com/anirudh2312/deep-learning/blob/master/images/Picture2.png)
Feature engineering is a broad topic and there are many subproblems that come under feature engineering, like Feature Extraction, Feature Selection and Feature Creation.

#### Feature Extraction: ‘automatic construction of new features from raw data’
Some observations, in their raw state are far too voluminous to be modelled by predictive modelling algorithms directly. Common examples include: image, audio, video, textual data. 
![Pic3](https://github.com/anirudh2312/deep-learning/blob/master/images/Capture.PNG)

Hence, Feature Extraction is a process of automatically reducing the dimensionality of these types of observations into a much smaller set that can be modelled. Methods include: Principal Component Analysis (PCA), clustering, line or edge detection. 

#### Feature Selection: ‘from many features to few that are useful’
In any machine learning problem, some features are redundant (maybe because they are too noisy or they are correlated to other features), or sometimes some features are much more important than others and we need to discard less important features because of the overfitting problem. We always tend to remove the attributes that are irrelevant to the problem. For this we apply feature selection techniques.

Feature selection can be done in varous ways:
##### Manual Feature Selection:
By observing the feature importance scores given by various algorithms like decision trees, etc.

##### Automated Feature Slection:
Recursive Feature Elimination: Select features by recursively considering smaller and smaller sets of features,
the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute.
The least important features are pruned from current set of features and this
procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

##### Regularization as Feature Selection (e.g. Lasso, Ridge):
They actively seek to remove or discount the contribution of features as part of the model building process

#### Feature Construction: ‘manual construction of new features from raw data’
1. Construction of new features requires spending a lot of time with actual sample data and thinking about the underlying form of the problem, structures in data and how best to expose them to predictive modelling algorithms
2. Feature creation requires great amount of domain knowledge
3. This is the part of feature engineering that is often talked the most about as an artform, the part that is attributed the importance and signalled as the differentiation in competitive machine learning
4. It is manual, it is slow, it requires lots of human brain power, and it makes a big difference!





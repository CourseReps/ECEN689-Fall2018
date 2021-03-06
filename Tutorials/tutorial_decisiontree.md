# Binary Decision Trees and Random Forests

#### by Vedant Mehta and Neehar Yalamarti

---

### What's a Decision Tree?

A Decision Tree Classifier, repetitively divides a feature space into sub parts by identifying splits repetitively. A Binary Decision Trees are called so because there is one split that occurs in each iteration which divides the feature space into two.
Decision trees can be applied to both regression and classification problems. A typical decision tree looks like this:

### Types of Decision Trees

There are various Decision tree algorithms which are different mainly based on splitting criterion and their application. Some are:

* CART
* ID3
* C4.5
* C5.0

### Advantages and Limitations of Decision Trees

#### Advantages

* They are simple to understand and interpret. Due to this, managers and stake holders prefer this algorithm.
* It handles numerical and qualitative data without explicitly encoding for variables.
* It makes no assumptions of the training data; e.g., distributional, independence or constant variance assumptions (typically done in linear regression)

#### Disadvantages

* They are typically not very robust. A small change in the training data can result in a large change in the tree (suffers from high variance)
* They can easily overfit data. Pruning is necessary to avoid this problem.

### Data

* Consider a simple two class problem with two features (as shown in the snippet below)
<p align="center">
<img src="https://github.com/CourseReps/ECEN689-Fall2018/blob/master/Students/vedantmehta2808/Figures%20for%20Tutorial/Data%20snippet.JPG" width="300" height="400">
</p>

And this is how the feature space looks like:

<p align="center">
  <img src="https://github.com/CourseReps/ECEN689-Fall2018/blob/master/Students/vedantmehta2808/Figures%20for%20Tutorial/FeatureSpace.png" width="350" height="250"></img>
</p>

The goal of decision tree is to divide the feature space into regions such that each of the regions have as much purity (data belonging to a particular class) as possible. So in order to divide the feature space, widely famous loss function named Gini Impurity is used. 

Gini Impurity is defined as below:
<p align="center">
  <img src="

### Prediction

**For a Classification problem:**
If the new data point lies in a particular region, then that point is predicted according to the associated highest probability of a particular class in that region.

**For a Regression problem:**
The new data point’s value is predicted as the mean of all the responses in that particular region.

### Pruning

To avoid building large trees which might overfit the data, pruning is done to reduce the size of the tree. There are several techniques to prune a tree, one of them being based on some criteria like minimum samples in each node, or minimum gini decrease.

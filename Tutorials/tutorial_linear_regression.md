{\rtf1\ansi\ansicpg1252\cocoartf1671
{\fonttbl\f0\fnil\fcharset0 HelveticaNeue-Medium;\f1\fnil\fcharset0 HelveticaNeue;\f2\fnil\fcharset0 Menlo-Regular;
}
{\colortbl;\red255\green255\blue255;\red25\green25\blue25;\red38\green38\blue38;\red242\green242\blue242;
}
{\*\expandedcolortbl;;\cssrgb\c12941\c12941\c12941;\cssrgb\c20000\c20000\c20000;\cssrgb\c96078\c96078\c96078;
}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}}
\paperw11900\paperh16840\margl1440\margr1440\vieww28600\viewh18000\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs72 \cf2 \expnd0\expndtw0\kerning0
ECEN 689 Tutorial\
\pard\pardeftab720\partightenfactor0

\fs48 \cf2 Samyuktha Sankaran, Harinath PRS\
\pard\pardeftab720\partightenfactor0

\fs60 \cf2 1 Abstract\
\pard\pardeftab720\partightenfactor0

\f1\fs28 \cf2 We had presented a tutorial on Linear regression. We explained the concept and the mathematics\
behind the model and demonstrated the same with a sample code. We had also discussed some\
possible applications and anomalies in the model.\
\pard\pardeftab720\partightenfactor0

\f0\fs60 \cf2 2 Linear Regression\
\pard\pardeftab720\partightenfactor0

\f1\fs28 \cf2 Linear Regression is a type of supervised model, used to find the relationship between the predictors\
and the target variables. There are two types of Linear regression: Simple and Multiple Linear\
regression.\
Simple Linear Regression is where a relationship is established between two quantities - one is\
the independent variable, that is the predictor and the dependent variable, the output. The idea is\
to obtain a line that best fits the data such a way that the total error is minimal.\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf3 \cb4 Yp=\uc0\u946  0 +\u946  1 X (1)\
error=\
\uc0\u8721 n\
i=\
(actual value\uc0\u8722 predicted value)^2 (2)\
\pard\pardeftab720\partightenfactor0

\f1\fs28 \cf2 \cb1 The error is squared, so that the positive and negative terms do not get cancelled. If\uc0\u946  1 > 0\
implies that a positive relation exists between the predictor and the target. And if\uc0\u946  1 < 0 , there is\
a negative relationship between the predictor and target.\
\pard\pardeftab720\partightenfactor0

\f0\fs48 \cf2 2.1 Metrics for Model evaluation\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0
\f1\fs28 \cf2 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
RSSThe Residual Sum of squares gives information about how far the regression line is from\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
the average of actual output.\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Sum of squared errortells how much the target values vary around the regression line.\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Total Sum of squaresis how much the data points are scattered about the mean.\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
P-Valuedescribes the relation between the null hypothesis and predicted value. A high P\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
value would mean that changes in the predictor have no effect on the target variable. A low\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
P value rejects the null hypothesis indicating that there is a relation between the target and\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
the predictor variables.\
\pard\pardeftab720\partightenfactor0

\f0\fs36 \cf2 1\
\pard\pardeftab720\partightenfactor0

\f1\fs28 \cf2 Multiple Linear regression attempts to model a relationship between two or more predictor\
variables and the target variable by fitting a linear equation to the observed data. Every value of\
the independent variables x are associated with the dependent variable y.\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf3 \cb4 Yp=\uc0\u946  0 +\u946  1 X 1 +\u946  2 X 2 +....+\u946 nXn (3)\
\pard\pardeftab720\partightenfactor0

\f1\fs28 \cf2 \cb1 An extension to the linear regression model is thepolynomial regression, where a non linear\
equation would be a best fit for the observed data set.\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf3 \cb4 Yp=\uc0\u946  0 +\u946  1 X 1 +\u946  2 X 22 (4)\
\pard\pardeftab720\partightenfactor0

\f0\fs60 \cf2 \cb1 3 Demonstration\
\pard\pardeftab720\partightenfactor0

\f1\fs28 \cf2 We demonstrated the working of a linear regression model through a small code snippet. We have\
used the mpg datasets that estimates the miles per gallon for automobiles based on parameters like\
number of cylinders, displacement, horsepower, power, weight, acceleration and model manufactured\
year.\
\pard\pardeftab720\partightenfactor0

\f0\fs48 \cf2 3.1 Implementation\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0
\f1\fs28 \cf2 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
After loading the train and test datasets, process the datasets by dropping entries that have\
\ls2\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
missing entries. One-hot encoding was applied to the origin of the automobile predictor\
\ls2\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
variable.\
\ls2\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Import the linear regression model from the SciKit learn package.\
\ls2\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Fit the training data to the model and print the coefficients of the respective predictor variable.\
\ls2\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
The coefficients comments about the effect the predictor has on the target variable.\
\ls2\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Pass the test data set and obtain the accuracy and mean squared error.\
\ls2\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Plot the obtained result}
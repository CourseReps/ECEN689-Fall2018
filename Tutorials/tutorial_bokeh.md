# Introduction to Bokeh

### by Khaled Nakhleh and Kishan Shah
### Course instructor: Jean-Francois Chamberland


Our presentation, presented on September 20th 2018, Introduced the Python packege called [**Bokeh**](https://bokeh.pydata.org/en/latest/).
Please be aware that we are not *Bokeh* experts, and this presentation is intended to entice students to start experimenting with *Bokeh* on their own time. We hope this presentation will explain the *Bokeh's* basics, importance, and capabilities. 

We should mention that our ideal steup includes a combination of *Bokeh* and *Matplotlib*. Using *Matplotlib* for static graphs (i.e. non-interactive graphs), and *Bokeh* for interactive visuals. Also, if you want to learn *Bokeh*, there will be an unavoidable learning curve to get familiarized with *Bokeh*. However, we believe that once you master *Bokeh*, you will become more efficient with presenting data visually.

### This Tutorial will cover the following:

1. *Bokeh* Breif description
2. Why use *Bokeh* over other packages?
3. Relevance to ECEN 689 course
4. *Bokeh* installation
5. *Bokeh* structure
6. Code examples
7. More reasons for using *Bokeh*
8. References for mastering *Bokeh*


#### 1. Brief introduction:

*Bokeh* is a visualization tool available for several programming languages: Python, R, Lua, and Julia. For the purpose of aligning with the course's objectives, we will only cover the Python *Bokeh* package. 

As mentioned, *Bokeh* is a tool for presenting data elegantly and interactively. *Bokeh* offers commands for manipulating the graph after running your script file using your web browser. *Bokeh* saves the graph as a .html file in your directory. *Bokeh* Provides elegant & concise way to construct versatile graphics while delivering high performance interactivity for large or streamed datasets.

The package *Bokeh* is a fiscally sponsored project of NumFOCUS, which supports other popular Python packages such as: Numpy, Pandas, and Matplotlib. *Bokeh* helps to create interactive plots, dashboards, and data applications efficiently.


#### 2. Why use *Bokeh*?:

The main reason we want students to start using *Bokeh* is to better represent the data structures they processed. Using only static graphs, student would need to type paragraphs to explain data's behavior. Not to mention the time and effort it takes to explain the technicalities of your graph to an audience. 

We must also, as engineers, assume our audience is not familiar with the topic we're presenting. The audience may come form different educational backgrounds, and may not be able to visually imagine theoratical concepts. Our presentattion should reflect the educational diversity of our audience. We believe with *Bokeh* this process would become much simpler to implement, and more visually approchable to the audience's entirety.  

#### 3. Relevance to ECEN 689 course:

Reading the course's syllabus, under course description, it reads as follows:
> The focus is on modular projects, algorithms and implementation, data management, and visualization. 

and under course objectives:
> Develop the ability to bridge theoretical concepts and practical tasks while dealing with information extraction. 

Based on the two above syllabus quotes, we picked *Bokeh* package to support the fulfillment 
of these objectives. Broadly speaking, the course also focuses on imporving students' Python skills, and students' presentation skills. Again, *Bokeh* is a Python package that builds upon what we're acheiving in this graduate course.

#### 4. *Bokeh* installation:

If you're using Python, we strongly recommend you use [Anaconda Python package manager](https://www.anaconda.com)

* Make sure you have *Bokeh* installed on your machine. To do that, type in the terminal:
``` 
conda list
```
* The above command will list all installed packages alphabetically. If *Bokeh* is not listed, you can install it using:
```
conda install bokeh
```
* The above command will install the latest version of *Bokeh* available on Anaconda's server.
* to update *Bokeh* to the lastest version, type the following into the terminal:
```
conda update bokeh
```
and if you want to update all packages, you can use:
```
conda update --all
```
* In case you're using the more traditional [**pip**](https://pypi.org/project/pip/), type:
```
pip install bokeh 
```
Follow these steps, and you should now have *Bokeh* installed on your machine. 

#### 5. *Bokeh* structure:

The *Bokeh* python package consists of three main modules:

**1) bokeh.charts**
This is the simplest of all *Bokeh* modules. **We don't recommend you use this package, since it's dated now and offers few customization options**. It offers basic functionality, and from what we read, it is no longer officially supported by the GitHub repository. 

**2) bokeh.plotting**

**3) bokeh.models**


#### 6. Code examples:

We wrote two basic Python script files to illustrate basic *Bokeh* functions. Code Snippets for the two examples are provided below. 

###### First example: graphing a line

```python
import numpy as np
from bokeh.plotting import *

"------------------------------------------------------------"

# Generate random data using numpy arrays.

x = np.arange(10)
y = np.random.randint(0, 20, size = 10)


# Make a title, x-axis label, and y-axis label.
plot = figure(x_axis_label = "X-axis", y_axis_label = "Y-axis", title = "Example 1")

# Plot command with attributes.
plot.line(x, y, line_color = "red", legend = "Random Y values") 

# Show the plot in browser (HTML format).
show(plot)
```

###### Second example: graphing circles and specifying tools used

```python
import numpy as np
from bokeh.plotting import *

"------------------------------------------------------------"

# Generating data using numpy arrays.
n = 1000

x = np.random.random(size = n) * 80
y = np.random.random(size = n) * 80
r = np.random.random(size = n) * 2

# Choosing graph tools
t = "pan, lasso_select, box_zoom, reset, tap, hover"

# Plot the figure with the listed tools in the string above.
plot = figure(x_range = (0, 100), y_range = (0, 100), tools = t,\
              x_axis_label = "X-axis", y_axis_label = "Y-axis", title = "Example 2")

# Generate circles on 2-D plane.
plot.circle(x, y, radius = r, fill_alpha = 0.3)

# Show the plot in browser (HTML format).
show(plot)
```

#### 7. More reasons for using *Bokeh*:



#### 8. References for mastering *Bokeh*

Here is a list of all sources we used to learn *Bokeh* and for the presentation.
We recommend you start with the official *Bokeh* manual:

* https://bokeh.pydata.org/en/latest/
* https://github.com/bokeh/bokeh
* https://numfocus.org/
* ECEN 689 Applied Information Science Practicum Fall 2018 Syllabus



We hope this tutorial benefited you. Good luck!

# Introduction to Bokeh

### by Khaled Nakhleh and Kishan Shah
### Course instructor: Prof. Jean-Francois Chamberland

---

Our presentation, presented on September 20th 2018, Introduced the Python packege called [**Bokeh**](https://bokeh.pydata.org/en/latest/).
Please be aware that we are not *Bokeh* experts, and this presentation is intended to entice students to start experimenting with *Bokeh* on their own free time. We hope this presentation will explain *Bokeh's* basics, importance, and capabilities. 

We should mention that our ideal Python setup includes a combination of *Bokeh* and *Matplotlib*. Using *Matplotlib* for static graphs (i.e. non-interactive graphs), and *Bokeh* for interactive visuals. Also, if you want to learn *Bokeh*, there will be an unavoidable learning curve to get familiarized with *Bokeh*. However, we believe that once you master *Bokeh*, you will become more efficient with presenting data visually.

### This Tutorial will cover the following

1. *Bokeh* Breif description
2. Why *Bokeh* over other packages?
3. Relevance to ECEN 689 course
4. *Bokeh* installation
5. *Bokeh* structure
6. Code examples
7. Reasons to use *Bokeh* and some limitations
8. Some fun and powerful examples that showcase *Bokeh's* capabilities
9. References for mastering *Bokeh*

---

### 1. Brief introduction

*Bokeh* is a visualization tool available for several programming languages: Python, R, Lua, and Julia. For the purpose of achieving courses' objectives, we will only cover the Python *Bokeh* package. 

As mentioned, *Bokeh* is a tool for presenting data elegantly that targets web browsers. *Bokeh* offers commands for manipulating the graph after running your script file using your web browser. *Bokeh* saves the graph as a .html file in your directory. *Bokeh* Provides elegant & concise way to construct versatile graphics while delivering high performance interactivity for large or streamed datasets.

The package *Bokeh* is a fiscally sponsored project of NumFOCUS, which supports other popular Python packages such as: Numpy, Pandas, and Matplotlib. *Bokeh* helps to create interactive plots, dashboards, and data applications efficiently.

*Bokeh* was created by Peter Wang, CTO and Co-founder of Continuum Analytics. Mr. Bryan Van de Ven is the lead developer as of September, 2018.

---

### 2. Why use *Bokeh*?

The main reason we want students to start using *Bokeh* is to better represent the data structures they processed. Using only static graphs, student would need to type text paragraphs to explain their data's behavior. Not to mention the time and effort it takes to explain the technicalities of your graph to an audience. 

Another reason to use *Bokeh* is that, as engineers, we must assume our audience is not familiar with the topic we're presenting. Our audience may come form different educational backgrounds, and may not be able to visually imagine theoratical concepts. Hence, our presentation should reflect the educational diversity of our audience. Using *Bokeh* simplifies our data implementations, and makes it visually approchable to the audience's entirety.  

*Bokeh* also:

* can transform other libraries' output like matplotlib, ggplot.
* Provides output to various mediums like html or jupyter notebook.
* can embed graphs to other services like django.

---

### 3. Relevance to ECEN 689 course

Under course description, it reads:
> The focus is on modular projects, algorithms and implementation, data management, and visualization. 

and under course objectives:
> Develop the ability to bridge theoretical concepts and practical tasks while dealing with information extraction. 

Based on these two above syllabus quotes, we picked the *Bokeh* package to support the fulfillment 
of these objectives. Broadly speaking, the course also focuses on imporving students' Python skills, and students' presentation skills. Again, *Bokeh* is a Python package that improves presentations' visual appeal.

---

### 4. *Bokeh* installation

If you're using Python, we strongly recommend you install [Anaconda Python package manager](https://www.anaconda.com).

* Make sure you have *Bokeh* installed on your machine. To check, type in the terminal:
``` 
conda list
```
* The above command will list all installed packages alphabetically. If *Bokeh* is not listed, you can install it using:
```
conda install bokeh
```
* The above command will install the version of *Bokeh* available on Anaconda's server (provided you have anaconda installed).
* to update *Bokeh* to the lastest version, type the following into the terminal:
```
conda update bokeh
```
and if you want to update all packages, you can use this command:
```
conda update --all
```
* In case you're using the default [**pip package manager**](https://pypi.org/project/pip/), type:
```
pip install bokeh 
```
Follow these steps, and you should now have *Bokeh* installed on your machine. All you have to do now is import *Bokeh* into your script file. This step is show with code examples (section 6).

---

### 5. *Bokeh* structure

The *Bokeh* python package consists of three main interfaces:

**1) bokeh.charts:**

High-level interface. This is the simplest of all *Bokeh* modules. **We don't recommend you use this package, since it's dated now and offers few customization options**. It offers basic functionality, and from what we read, it is no longer officially supported in the GitHub repository. Think of it as a Microsoft Word equivalent charting tool.

**2) bokeh.plotting**

Intermediate-level interface. **This module is what we used and strongly recommend everyone to start with**. This package offers a nice balance between customizability and familiarity. 

**3) bokeh.models**

Low-level interface. The most complex of all three interfaces. This one is intended for expert *Bokeh* users. We think that once you have mastered the functions and tools in bokeh.plotting, you should start using this interface. This interface offers the greatest flexibility and options for visually presenting your data.

---

### 6. Code examples

We wrote two basic Python script files to illustrate basic *Bokeh* functions. Script files for the two examples are provided below. Please note that the line ```from bokeh.plotting import *``` imports all functions and classes in bokeh.plotting. If you want to reduce memory usage, then only import the functions you'll use.

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

Try running these examples on your local machine, and see what results you'll get.

---

### 7. Reasons to use *Bokeh* and some limitations

If you're planning on joining an industry, then you can take advantage of *Bokeh's* estabilished industry presence.
The Python *Bokeh* is actively used by the following institutions:

* PepsiCo
* Facebook
* Uber
* Spotify
* NBC universal
* and more!

**Limitations**
However, there are some limitations with *Bokeh*. Since *Bokeh* is relativity new, changes are being committed rapidly to the source code. This means that some concepts you learn may end up becoming obsolete in the next update. This is something all users should be aware of, not just for *Bokeh*, but for Python in general. 

The second limitation is the existance of *matplotlib* animation commands. It would make sense for novice programmers to start with *matplotlib* animation features, instead of learning a whole new package. However, we believe that once you step over the learning curve, *Bokeh* presents itself as a much more useful tool. We also think mastering *Bokeh* will help you become more efficient with visualizing large datasets.

---

### 8. Some fun examples that showcase *Bokeh's* capabilities 

These examples show how powerful this package is. They cover topics such as EM-signals and population graphs.
We encourage you to view these interactive graphs, and understand the available source code. 

* [3-D animated signal](https://demo.bokehplots.com/apps/surface3d)

* [Countries' life expectancy vs. fertility rate](https://demo.bokehplots.com/apps/gapminder)

* [Interactive graph](https://demo.bokehplots.com/apps/crossfilter)

Other useful examples can be found here: https://demo.bokehplots.com

---

### 9. References for mastering *Bokeh*

Here is a list of all the sources we used to learn about *Bokeh* and for the presentation.
We recommend you start with the official *Bokeh* manual:

* https://bokeh.pydata.org/en/latest/
* https://github.com/bokeh/bokeh
* https://numfocus.org/
* ECEN 689 Applied Information Science Practicum Fall 2018 Syllabus

We hope this tutorial helped you become a better Python programmer. Good luck with your projects!

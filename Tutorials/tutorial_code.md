# Production Code
by Aliakbar Shafi and Rishi Laddha
 
 
Programming is a very important facet for a data scientist. It runs parallel to all the other processes in any data science project; right from data acquisition, data cleaning, visualization, data modelling in development stage before going into deployment. Thus, being able to write a production-level code is a must for every data scientist. It not only helps in easy deployment but also makes maintenance and reusability much simpler. This tutorial aims to highlight some basic code practices and etiquettes who intends to be an awesome programmer and data scientist!
 
**1.	Modularity**
Breaking your code into modules makes it very easy to identify the functionality of any software. Also, modular approach helps in making the code extremely reusable and easy to maintain. 
 
*How can we do it?*
 
Step 1.  Break the code into smaller pieces each intended to perform a specific task.
 
Decompose a large code into simple functions with inputs and outputs in correct formats. Each function should perform a single task like removing outliers, correcting wrong data, calculating errors (MSE, RMSE, etc.). Each of the functions should be broken down to a level where they cannot be decomposed any further. 
 
The granularity of function decomposition can be divided in three parts. 
 
(i)	*Low-level functions* — the most basic functions that cannot be further decomposed. These are the functions which would find place in any data science task, example: computing MSE or Z score of data.
 
(ii) *Medium-level functions* — a function that uses one or more low-level functions and/or other medium-level functions to perform the task. For example, remove outliers function would use compute Z-score function to remove the outliers by only retaining data within certain bounds or an error function that uses RMSE function.
 
(iii) *High-level functions* — a function that uses one or more of medium-level functions and/or low-level functions to perform its task. For example, model training function can use several functions such as function to get randomly sampled data, a model scoring function, a metric function, etc.
 
 
Step 2.  Group these functions into modules (Jupyter notebooks/python files) based on its usability. 
 
Once the functions have been developed for all the tasks it is time to group all the low-level and medium-level functions that will be useful for more than one ML algorithm into a python file (a module to be imported) and all other low-level and medium-level functions that will be useful only for the algorithm in consideration into another module( python file). All the high-level functions should reside in a separate python file. The file with all the high level functions decides the algorithm development: right from combining data from different sources to final machine learning models. 
 
**2. Logging and Instrumentation**
Logging and Instrumentation [LI] record all the useful information happening in the code during its execution. LI helps the programmer to debug if anything does not work as expected  and to improve the performance of the code in terms of time and space complexity.
 
Logging vs Instrumentation
 
*Logging* : Records information such as critical failures during run time and intermediate results that will be used by the code itself. There are multiple log levels: notset, debug, info, warning, error, critical. Depending upon the severity of information we need, we can set the logging level. Although very helpful to a developer logging should be avoided as much possible during production stage and should contain only information that requires immediate human intervention.
 
*Instrumentation* : Records information that is not included in logging. Instrumentations helps us to validate the code execution steps and work on improvements if necessary. 
 
01.	 To validate code execution steps — Information such as task name, intermediate results etc should be recorded. Incorrect results or vague algorithm may not raise a critical error that would be caught in logging. Hence it is necessary to get this information.
02.	 To improve performance —  It is imperative to record time taken for each execution and memory occupied by each variable. We have a limited time and computing power. Instrumentation thus helps us in identifying the bottlenecks in our code and helps to optimize the code in terms of time and space complexity.
 
**3. Code Optimization**
As mentioned above, code optimization includes reducing time complexity as well as space complexity. Lower the time and space complexity more efficient is the algorithm.
 
Let’s say we have a nested for loop of size n each and takes about 2 seconds each run followed by a simple for loop that takes 3 seconds for each run. Then the equation for time consumption can be written as
Time taken ~ 2n²+3n = O(n²+n) = O(n²)
The for loops should be replaced with python modules or functions. These functions can be optimized using software written in any procedural language (C) instead of Object oriented (python) which would reduce the run time.
 
Eliminating nested loops is the most basic way towards optimizing codes. For better programming practices one should include important data structure practices in the code like: Stacks, queues, lists, binary trees, Graphs, etc. 
 
**4. Unit Testing**
Unit testing is used to automate code testing in terms of functionality. The code should be tested against different corner cases, different data sets and different situations. In a massive code, whenever we make any change it is inefficient to leave out the quality test for QA team and thus every data scientist should practice creating a Unit testing module containing set of test cases which can be executed whenever we want to test the code.
Unit testing module contains test cases with expected results to test the code. It goes through every test case, compares actual output with every expected output and fails if any discrepancy.  
Python has a module called unittest to implement unit testing.
 
**5. Readability**
The most important of all, any piece of software code should be written in such a way that it can easily read and understood not only by the data scientist but by anyone who has a limited understanding of programming language.Some practices that increase the readability of code are:
 
(i) *Appropriate variable and function names*
        “Sometimes it’s all in the name!” Someone reading your code should get an idea what is the piece of code trying to achieve through the variables and function names at least to some extent.
For example, the variable for average age of female in a sample data can be written as meanAgeFemale rather than age or x. The function names should also be named in a similar manner. The function names should be active verb form [calculateError(...)] and the variable names should be a noun 
 
(ii) *Doc string and comments*
         Just the variable and function names do not help in understanding the code, hence it is necessary to have comments wherever necessary to help the reviewer of code.In addition to appropriate variable and function names, it is essential to have comments and notes wherever necessary to help the reader in understanding the code.
 
*Doc string (Function/class/module specific)*: Theses are lines of text inside the function definition that describe the function role, inputs and outputs. The text should be placed between set of 3 double quotes.
 
def <function_name>:
“””<docstring>”””
return <output>
 
*Comments* —texts placed in the code to explain the reader about the a particular section/line. 
 
 
**6. Compatibility with ecosystem**
The code of a data scientist is not going to be a standalone piece of software. It will be integrated into organization’s existing framework/code system and the code has to run synchronously with other parts of the ecosystem without any disruptions across the program flow.
 
 
For example, let’s say as a Data Scientist you have been given a job to design a recommendation system. The system flow would be getting data from the database, generate recommendations, store it in a database. The new data is fetched by front-end frameworks (APIs/ Web- services) to display the recommended items to the user. 
 
Every step will have a expected input and output, response time, etc. Whenever any module requests for recommendations the application/website should return the expected values in a desired format in an acceptable time. If the results are unexpected values, undesired format or delayed time, the code is unsynchronized with system.
 
**7. Version Control**
Every time we alter the code, instead of saving the file with a different name, we commit the changes by overwriting the old file with new changes having a unique key associated with each version. If the new changes disrupt the process flow or do not meet the functional/ technical requirements, we can restore the old file by using the commit reference key of desired version. 
 
Even in a team with handful of programmers, it can become really difficult to manage development or testing. With Version control, it becomes extremely easy to keep the whole team on a same page.
 
Github and Subversion(SVN) are two most popular Version control softwares.

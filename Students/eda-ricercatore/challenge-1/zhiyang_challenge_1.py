#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

"""
	This is written by Zhiyang Ong to estimate the mean and variance
        for a series of data sets.

    That is, determine the mean and variance for each set of 12 samples.


	Synopsis: command name and [argument(s)]
	./zhiyang_challenge_1.py

	Parameters:
	[input BibTeX file]:	A BibTeX database.

	Its procedure is described as follows:
    Set up file I/O operations.
	Read the first line to get the information for the data set.
        Copy line to output file.
    For each subsequent line in the data set,
        split the line into tokens using "," as the delimiter.
        calculate the mean of the samples
        calculate the variance of the samples
        Write modified line to output file.
    Close file I/O operations.

	Notes/Assumptions:
	[To be completed]



	References:
	Citations/References that use the LaTeX/BibTeX notation are taken
    	from my *BibTeX* database (set of BibTeX entries).



	Revision History:
	September 4, 2018			Version 0.1, initial build.
"""


#	The MIT License (MIT)

#	Copyright (c) <2014-2017> <Zhiyang Ong>

#	Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#	The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#	Email address: echo "cukj -wb- 23wU4X5M589 TROJANS cqkH wiuz2y 0f Mw Stanford" | awk '{ sub("23wU4X5M589","F.d_c_b. ") sub("Stanford","d0mA1n"); print $5, $2, $8; for (i=1; i<=1; i++) print "6\b"; print $9, $7, $6 }' | sed y/kqcbuHwM62z/gnotrzadqmC/ | tr 'q' ' ' | tr -d [:cntrl:] | tr -d 'ir' | tr y "\n"	Che cosa significa?

#	==========================================================

__author__ = 'Zhiyang Ong'
__version__ = '0.1'
__date__ = 'Sep 4, 2018'

###############################################################

"""
	Import modules from The Python Standard Library.
	sys			Get access to any command-line arguments.
	os			Use any operating system dependent functionality.
	os.path		For pathname manipulations.

	subprocess -> call
				To make system calls.
	time		To measure elapsed time.
	warnings	Raise warnings.
	re			Use regular expressions.
	filecmp		For file comparison.
"""

import sys
import os
import os.path
from subprocess import call
import time
import warnings
import re
import filecmp
from datetime import date
import datetime
import pandas as pd
import numpy as np
#from scipy import stats
from scipy.stats import norm

###############################################################
#	Import Custom Python Packages and Modules

# Package and module to perform file I/O (input/output) operations.
from utilities.file_io import file_io_operations

###############################################################
class mean_variance_estimator:
	# Filename of the input file:
	input_filename = "1challenge1activity_zhiyang.csv"
	# Filename of the output file:
	output_filename = "1challenge1activity_zhiyang_ong.csv"
	# File object for input processing.
	ip_file_obj = None
	#ip_file_obj = file_io_operations.open_file_object_read("=")
	# File object for output processing.
	op_file_obj = None
	#op_file_obj = file_io_operations.open_file_object_write("=")
	# Size of each sample set
	sample_set_size = 12.0
	# First line of the series of sample sets
	first_line = "Set number,Mean,Variance,Sample 0,Sample 1,Sample 2,Sample 3,Sample 4,Sample 5,Sample 6,Sample 7,Sample 8,Sample 9,Sample 10,Sample 11\n"
	##	Method for preprocessing.
	#	O(1) method.
	@staticmethod
	def preprocessing():
		"""
		if ip_file_obj is not None:
			file_io_operations.close_file_object(ip_file_obj)
		if op_file_obj is not None:
			file_io_operations.close_file_object(op_file_obj)
		"""
		print("=	Create a file object for reading and writing.")
		# Create a file object for input BibTeX file, in reading mode.
		mean_variance_estimator.ip_file_obj = file_io_operations.open_file_object_read(mean_variance_estimator.input_filename)
		mean_variance_estimator.op_file_obj = file_io_operations.open_file_object_write_new(mean_variance_estimator.output_filename)
	##	Method for postprocessing.
	#	O(1) method.
	@staticmethod
	def postprocessing():
		print("=	Close the I/O file objects.")
		if mean_variance_estimator.ip_file_obj is not None:
			file_io_operations.close_file_object(mean_variance_estimator.ip_file_obj)
		if mean_variance_estimator.op_file_obj is not None:
			file_io_operations.close_file_object(mean_variance_estimator.op_file_obj)
	##	Method for obtaining the ndarray object (from the NumPy’s array
	#		class) for a given list.
	#	@param py_list - A Python list of numbers.
	#	@return ndarray object for py_list.
	#	O(1) method, for a list.
	@staticmethod
	def get_ndarray_obj_for_list(py_list):
		return np.array(py_list)
	##	Method for sample mean calculation.
	#	O(n) method, for number of samples.
	@staticmethod
	def get_sample_mean(sample_set):
		return sum(sample_set)/sample_set_size
	##	Method for sample mean estimation.
	#	@param ndarray_obj - A ndarray object.
	#	@return mean estimation for ndarray object.
	#	O(n) method, for number of samples.
	@staticmethod
	def get_mean_estimation(ndarray_obj):
		return np.mean(ndarray_obj)
	##	Method for sample variance estimation.
	#	@param ndarray_obj - A ndarray object.
	#	@return variance estimation for ndarray object.
	#	O(n) method, for number of samples.
	@staticmethod
	def get_variance_estimation(ndarray_obj):
		return np.var(ndarray_obj)
	"""
		Develop Python method with multiple return values.
		Return set number, and list of samples.
	"""
    ##	Method to enumerate the file line by line.
	#	@return a boolean TRUE, if the path to the file is valid.
	#		Else, return FALSE.
	#	O(1) method.
	@staticmethod
	def enumerate_file():
		print("=	Enumerate each line in the input file.")
		for line in mean_variance_estimator.ip_file_obj:
			if line == mean_variance_estimator.first_line:
				mean_variance_estimator.op_file_obj.write(line)
				#mean_variance_estimator.op_file_obj.write("Finished printing line #1.\n")
			else:
				mean_variance_estimator.op_file_obj.write(line)
###############################################################
# Main method for the program.

#	If this is executed as a Python script,
if __name__ == "__main__":
	print("===================================================")
	print("Estimate mean and variance for a series of data sets.")
	print("")
	mean_variance_estimator.preprocessing()
	mean_variance_estimator.enumerate_file()
	mean_variance_estimator.postprocessing()
	print("	= end =")

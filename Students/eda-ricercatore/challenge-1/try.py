#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

"""
	This is written by Zhiyang Ong to estimate the mean and variance
        for a data set.

	Synopsis: command name and [argument(s)]
	./try.py

	Parameters:
	[input BibTeX file]:	A BibTeX database.

	[-h]				:	If an optional "-h" flag is used as an
							input argument, show the brief user manual
							and exit (terminate the program).

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
# Main method for the program.

#	If this is executed as a Python script,
if __name__ == "__main__":
	print("===================================================")
	print("Estimate mean and variance for a series of data sets.")
	print("")
	a = [-0.7121442845649851,-0.5900880509782601,0.5565013031553446,0.6972370680510693,0.03009375657100899,0.8444878494671966,0.27872272499591866,-0.11401063556351954,0.4981329984008711,1.1179404389357728,0.5710573787587523,-0.8843705649395984]
	b = np.array(a)
	print("	Estimated mean is:",np.mean(b),"=")
	print("	Estimated variance is:",np.var(b),"=")
	
	print("	= end =")

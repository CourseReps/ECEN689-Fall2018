#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep  4 

Spyder Editor
Task 6 exercises 

Name: Khaled Jamal Nakhleh
M.S. Electrical Engineering.
Code was run on the website. I also copied it here.

I DID NOT CREATE NOR DO I OWN THIS CODE. 
THIS FILE WAS ONLY MADE TO COMPLETE COURSE 689 TASK 6.

THE CODE IS FROM GOOGLE COLABORATORY WEBSITE:
    https://colab.research.google.com/notebooks/welcome.ipynb
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2
#Executing Tensorflow

input_1 = tf.ones((2, 3))
input_2 = tf.reshape(tf.range(1, 7, dtype=tf.float32), (2, 3))

output = input_1 + input_2

with tf.Session():
    result = output.eval()


x = np.arange(20)
y = [x_i + np.random.randn(1) for x_i in x]
a, b = np.polyfit(x, y, 1)

p_1 = plt.plot(x, y, 'o', np.arange(20), a*np.arange(20)+b, '-')
plt.show()

p_2 = venn2(subsets = (3, 2, 1))
plt.show()


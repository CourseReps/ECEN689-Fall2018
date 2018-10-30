#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Sep  7 20:28:50 2018

Name: khalednakhleh

Acknowledgments:

This file was written with help from Siladittya Manna's instructions
on parsing idx files in Python. Under the MIT license, his work can be modified
and redistributed, as long as credit was given.

Please check his GitHub page for idx file format-reading in Python:
https://github.com/sadimanna/idx2numpy_array

"""

""" Class file decodes an idx formatted file into a numpy array (predictor-readable data). """

import numpy as np
import struct as st


class De_idx_set:
    
    def __init__(self,file_name):
        """ Initiates decoding idx file for instance defined. """
        
        self.file_name = str(file_name)
        self.idx_file = open(self.file_name, "rb")
        self.bin_data = self.idx_file.read()
        self.magic_no, self.item_no, self.row_no, self.col_no = self.unpack()
        self.array = self.array_generation()
        
    def unpack(self):
        """ Unpacks header info in idx file's first 15 bytes. """
        
        self.idx_file.seek(0)
        magic_no = st.unpack(">4B", self.idx_file.read(4))
        item_no = st.unpack(">I", self.idx_file.read(4))[0]
        row_no = st.unpack(">I", self.idx_file.read(4))[0]
        col_no = st.unpack(">I", self.idx_file.read(4))[0]
        
        return magic_no, item_no, row_no, col_no
    
    def array_generation(self):
        """ Generates a numpy array from the 16th bytes onwards. """
        
        byte_no = 1 * self.item_no * self.col_no * self.row_no
        dimension = (self.item_no, self.col_no * self.row_no)
        
        array = 255 - np.asarray(st.unpack('>'+'B'*byte_no,(self.idx_file).read(byte_no)))\
                .reshape(dimension)
        
        return array

class De_idx_label:
    
    def __init__(self,file_name):
        """ Initiates decoding idx file for instance defined. """
        
        self.file_name = str(file_name)
        self.idx_file = open(self.file_name, "rb")
        self.bin_data = self.idx_file.read()
        self.magic_no, self.label_no = self.unpack()
        self.array = self.array_generation()
        
    def unpack(self):
        """ Unpacks header info in idx file's first 15 bytes. """
        
        self.idx_file.seek(0)
        magic_no = st.unpack(">4B", self.idx_file.read(4))
        label_no = st.unpack(">I", self.idx_file.read(4))[0]
        
        return magic_no, label_no
    
    def array_generation(self):
        """ Generates a numpy label array. """
        # Add 255 - at the beginning if binary is required
        array = np.asarray(st.unpack('>'+'B'*self.label_no,(self.idx_file).read(self.label_no)))

        return array


if __name__ == "__main__":
    
    print("\n\nThis file is intended as a Class file only. Please use main.py\n\n")
    
    exit

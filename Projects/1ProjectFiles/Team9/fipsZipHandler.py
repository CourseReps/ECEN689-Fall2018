# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:02:19 2018

@author: Harish K
"""

import numpy as np


class FipsZipHandler:
    dataFilePrefix = 'datasets/'
    fipsToZipFile = 'fipsToZip.csv'
    zipToFipsFile = 'zipToFips.csv'
    fipsToNameAndStateFile = 'fipsToNameAndState.csv'
    
    def __init__(self):
        fipsToZipMap, zipToFipsMap, fipsToCountyNameAndStateMap = self.__prepareAndGetDictionaries()
        self.fipsToZipMap = fipsToZipMap
        self.zipToFipsMap = zipToFipsMap
        self.fipsToCountyNameAndStateMap = fipsToCountyNameAndStateMap
        
    def __loadAndGetData(self):
        fipsToZip = np.loadtxt(self.dataFilePrefix + self.fipsToZipFile, 
                               dtype='str', delimiter=',')
        zipToFips = np.loadtxt(self.dataFilePrefix + self.zipToFipsFile,
                               dtype='str', delimiter=',')
        fipsToNameAndState = np.loadtxt(
                self.dataFilePrefix + self.fipsToNameAndStateFile, 
                                        dtype='str', delimiter=',')
        return fipsToZip, zipToFips, fipsToNameAndState
    
    def __getDictionaryFromNumpyStringArray(self, npArray):
        result = {}
        for i in range(len(npArray)):
            key = npArray[i,0]
            value = []
            for j in range(1, len(npArray[i])):
                value += [npArray[i][j]]
            if key not in result.keys():
                result[key] = value
            else:
                result[key] = result[key] + value
        return result
    
    def __prepareAndGetDictionaries(self, ):
        fipsToZip, zipToFips, fipsToNameAndState = self.__loadAndGetData()
        fipsToZipsMap = self.__getDictionaryFromNumpyStringArray(fipsToZip)
        zipToFipsMap = self.__getDictionaryFromNumpyStringArray(zipToFips)
        fipsToCountyNameAndStateMap = self.__getDictionaryFromNumpyStringArray(
                fipsToNameAndState)
        return fipsToZipsMap, zipToFipsMap, fipsToCountyNameAndStateMap
    
    def getFipsForZipcode(self, zipcode): #Returns a string containing the fips
        # Return NaN if the zip code isn't in the map.
        if zipcode not in self.zipToFipsMap.keys():
            return np.NaN
        # Look up and return zip code
        return self.zipToFipsMap[zipcode][0]
        
    def getZipcodesForFips(self, fips): #Returns a list containing all the zips
        #in the county
        if fips not in self.fipsToZipMap.keys():
            return np.NaN
        return self.fipsToZipMap[fips]
    
    def getCountyNameAndStateForFips(self, fips): #Returns a map containing
        #The state and the county for a fips
        if fips not in self.fipsToCountyNameAndStateMap.keys():
            return np.NaN
        return {'state': self.fipsToCountyNameAndStateMap[fips][1], 
                'county': self.fipsToCountyNameAndStateMap[fips][0]}
    
    def getCountyNameAndStateForZip(self, zipcode): #Returns a map containing 
        #The state and the county for a zip
        return self.getCountyNameAndStateForFips(
                self.getFipsForZipcode(zipcode))

if __name__ == '__main__':
    obj = FipsZipHandler()
    pass

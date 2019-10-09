#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:31:58 2019

@author: miguel
"""
count = []
a = []
with open('names.txt','r') as names:
    for linea in names:
        linea = str(linea)
        linea = linea.rstrip('\n')
        a=a.append(linea)
       # count = count.append(user.count(linea))
        

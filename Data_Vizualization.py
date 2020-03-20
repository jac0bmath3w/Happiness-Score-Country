#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:00:25 2020
Visualization of the World Happiness Scores Data
@author: jacob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

data = pd.read_csv('World_Happiness_2015_2017.csv')
# Then what I do next is look into shape using data.shape(). This will tell me how many rows and columns there are.
data.shape

g = sns.pairplot(data)
g.fig.suptitle('FacetGrid plot', fontsize = 20)
g.fig.subplots_adjust(top= 0.9);


# Creating a list of attributes we want (just copy the column name)
econ_happiness = ['Happiness Score','Economy (GDP per Capita)']

# Creating a dataframe that only contains these attributes
econ_corr = data[econ_happiness]

# Finding correlation
econ_corr.corr()
sns.regplot(data = econ_corr, x = 'Economy (GDP per Capita)', y = 'Happiness Score').set_title("Correlation graph for Happiness score vs Economy")
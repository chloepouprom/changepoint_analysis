
# coding: utf-8

# In[4]:


import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os
import glob
import ludwig
import re
from scipy import stats
from time_series_analysis import TimeSeriesAnalysis
import csv


# In[5]:


# SESSION1_AU_FILES = glob.glob('/Volumes/chloe/ROBOT_DATA/Session1/*/AU_session*')
# SESSION2_AU_FILES = glob.glob('/Volumes/chloe/ROBOT_DATA/Session2/*/AU_session*')
# SESSION3_AU_FILES = glob.glob('/Volumes/chloe/ROBOT_DATA/Session3/*/AU_session*')

# OUTPUT_DIRECTORY = './logs'

SESSION1_AU_FILES = glob.glob('/u/chloe/ROBOT_DATA/Session1/*/AU_session*')
SESSION2_AU_FILES = glob.glob('/u/chloe/ROBOT_DATA/Session2/*/AU_session*')
SESSION3_AU_FILES = glob.glob('/u/chloe/ROBOT_DATA/Session3/*/AU_session*')

OUTPUT_DIRECTORY = './logs/'


# In[6]:


def get_changepoints(au_file):    
    [lip_raiser_data, jaw_lower_data, lip_stretcher_data,      brow_lower_data, lip_corner_depressor_data, brow_raiser_data,      au_dates] = ludwig.extract_animation_units(au_file)

    changepoints_per_au = {'lr': [], 'jl': [], 'ls': [], 'bl': [], 'lcd': [], 'br': []}
    for i, data in enumerate([lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data]):
        filtered_data = scipy.signal.medfilt(data, 101)
        ts = TimeSeriesAnalysis(au_dates, filtered_data)
        ts.ChangePointAnalysis()
        changepoints = [el[0] for el in ts.changepoints]
        changepoints = [au_dates[cp] for cp in changepoints]
        changepoints_per_au[list(changepoints_per_au.keys())[i]] = changepoints

    return changepoints_per_au

for au_file in SESSION1_AU_FILES + SESSION2_AU_FILES + SESSION3_AU_FILES:
    cps = get_changepoints(au_file)
    output_filename = OUTPUT_DIRECTORY + au_file.replace('/', '_')
    print(au_file)
    print(output_filename)    
    with open(output_filename, 'w') as f:
        for au in cps:
            str_cp = ''
            for cp in cps[au]:
                str_cp += str(cp) + ' '
            f.write('%s %s\n' % (au, str_cp))

    


# Importing the toolbox (takes several seconds)
import warnings

import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from textwrap import wrap
from scipy import integrate

warnings.filterwarnings('ignore')

all_data = pd.DataFrame()
list_no = np.arange(0.0, 108060.0, 1.0) #number of frames in 30 minutes
# list_no = np.arange(0.0, 180000, 1.0) #number of frames in 50 minutes
ms_time = np.arange(0.0, 2670.0, 0.4) #1ms increments of time
# frame rate of camera in those experiments
start_frame = 120 #frame to start at
pick_frame = 30 # pick every __th frame

fps = 60
no_seconds = 30


DLCscorer = 'DLC_resnet50_BigBinTopSep17shuffle1_250000'


def orientation(video):

    dataname = str(Path(video).stem) + DLCscorer + '_skeleton.h5'
    print(dataname)

    #loading output of DLC
    Dataframe = pd.read_hdf(os.path.join(dataname), errors='ignore')
    # Dataframe.reset_index(drop=True)

    #you can read out the header to get body part names!
    bodyparts=Dataframe.columns.get_level_values(1)

    bodyparts2plot=bodyparts

    # let's calculate velocity of the back
    # this can be changed to whatever body part
    bpt='back_tail_base'

    data = pd.DataFrame(data=None)
    data['length'] = Dataframe[bpt]['length'].values
    raw_angles =Dataframe[bpt]['orientation'].values
    data['orientation_angle'] = raw_angles
    raw_angles = pd.Series(raw_angles)
    orientation_avg = raw_angles.rolling(fps*no_seconds).mean()
    time = np.arange(len(data['orientation_angle'])) * 1. / fps  # time that is 1/60 sec
    data['time'] = time / 60



    query = video
    stopwords = ['top', 'down', 'top down']
    querywords = query.split('_')
    resultwords = [word for word in querywords if word.lower() not in stopwords]
    legend_name = ' '.join(resultwords)

    all_data[legend_name+'_time'] = data['time']
    all_data[legend_name+'_orientation'] = orientation_avg
    # plt.plot(data['time'], orientation_avg, label=legend_name)

if __name__ == '__main__':
    print("Hi")
    orientation(video='Saline_Ai14_OPRK1_C2_F1_Top_Down')
    orientation(video='U50_Ai14_OPRK1_C1_F1_Top Down')
    orientation(video='NORBNI_Saline_Ai14_OPRK1_C2_F1_Top Down')
    orientation(video='NORBNI_U50_Ai14_OPRK1_C2_F1_Top Down')
    orientation(video='Nalt_U50_Ai14_OPRK1_C1_F1_Top Down')





    # saline avg
    saline_values = all_data.loc[:,[
        'Saline Ai14 OPRK1 C2 F1_orientation'
    ]]
    saline_mean = saline_values.mean(axis=1)
    saline_time = (saline_mean.index) * (1. / fps) / 60
    saline_sem = stats.sem(saline_mean)

    plt.plot(saline_time, saline_mean)
    leg = plt.legend(loc='upper right', fontsize=12, frameon=False)
    plt.show()
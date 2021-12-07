#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:47:38 2019

@author: shane
"""
import os
import pandas as pd
import numpy as np

################################################################################
    # humidity data class
class HumidityData:
    
    def __init__(self, stn_numbers, wind_dir, weights):
        """
            Reads in the humidity data from the wind data file 
            
            Parameters
            ----------
                    stn_number : float
                    The weather station number to get the wind data for.
                    
                    wind_dir : string
                    Location of the wind data files.
                    
            Returns
            -------
                    None
        """
        self.stn_numbers = stn_numbers
        self.wind_dir = wind_dir
        
        all_hums = pd.DataFrame()
        
        for stn_number, weight in zip(stn_numbers, weights):
            #######################################################################
            # wind data, the spd_bins returned here are the bins that will be
            # used for the seasonal windrose
            wind_file = os.path.join(self.wind_dir,
                                     'wind_data_stn_{}.csv'.format(stn_number)
                                     )
            
            curr_hum = self.get_humidity(wind_file)
            curr_hum['weight'] = np.repeat(weight, curr_hum.shape[0])
            
            if all_hums.empty:
                all_hums = curr_hum
            else:
                all_hums = all_hums.append(curr_hum)

        self.humidity = all_hums
    
    def get_humidity(self, wind_file):
        """
        """
        df = pd.read_csv(wind_file, dtype=str)
        
        df_9am = df[['9am_temp', 
                     '9am_hum']].dropna().astype('float')
        
        df_9am.rename(index=str, 
                      columns={"9am_temp": "temp",
                               "9am_hum": "humidity"},
                               inplace=True
                      )
        
        df_3pm = df[['3pm_temp',
                     '3pm_hum']].dropna().astype('float')
        
        df_3pm.rename(index=str, 
                      columns={"3pm_temp": "temp",
                               "3pm_hum": "humidity"},
                               inplace=True
                      )
        
        return self.merge_humidity_dfs(df_9am, df_3pm)
            
    def merge_humidity_dfs(self, df1, df2):
        """
        """
        temps = np.append(df1['temp'].values,
                          df2['temp'].values)
        
        humidity = np.append(df1['humidity'].values,
                             df2['humidity'].values)
        
        return pd.DataFrame({'temp': temps, 
                             'humidity': humidity}) 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:14:05 2019

@author: shane
"""

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
class CloudData:
    
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
        self.wind_dir = wind_dir
        
        all_clouds = pd.DataFrame()
        
        for stn_number, weight in zip(stn_numbers, weights):
            #######################################################################
            # wind data, the spd_bins returned here are the bins that will be
            # used for the seasonal windrose
            wind_file = os.path.join(self.wind_dir,
                                     'wind_data_stn_{}.csv'.format(stn_number)
                                     )
            
            curr_clouds = self.get_clouds(wind_file)
            curr_clouds.clouds *= weight
            
            if all_clouds.empty:
                all_clouds = curr_clouds
            else:
                all_clouds = all_clouds.append(curr_clouds)

        all_clouds['Month'] = [int(x.split('-')[1]) for x in all_clouds.Date.values]

        self.clouds = (all_clouds.groupby('Month').clouds.sum() / np.sum(weights)) / \
                       all_clouds.groupby('Month').clouds.count()
                       
        self.clouds_std = all_clouds.groupby('Month').clouds.std() / np.sum(weights)

        self.clouds = pd.DataFrame({'Mean': self.clouds.values, 'Std': self.clouds_std.values},
                                   index=self.clouds.index)

    
    def get_clouds(self, wind_file):
        """
        """
        df = pd.read_csv(wind_file, dtype=str)
        
        df_9am = df[['Date', 
                     '9am_cloud']].dropna()
        
        df_9am.rename(index=str, 
                      columns={"Date": "Date",
                               "9am_cloud": "clouds"},
                               inplace=True
                      )
        
        df_3pm = df[['Date',
                     '3pm_cloud']].dropna()
        
        df_3pm.rename(index=str, 
                      columns={"Date": "Date",
                               "3pm_cloud": "clouds"},
                               inplace=True
                      )
        
        return self.merge_cloud_dfs(df_9am, df_3pm)
            
    def merge_cloud_dfs(self, df1, df2):
        """
        """
        Date = np.append(df1['Date'].values,
                         df2['Date'].values)
        
        clouds = np.append(df1['clouds'].values,
                           df2['clouds'].values).astype('float')
        
        return pd.DataFrame({'Date': Date, 
                             'clouds': clouds}) 
            

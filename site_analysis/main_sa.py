#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# standard libs
import os
import sys
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager

helv_font = 'add_files/fonts/HelveticaNeueLTStd-Th.otf'
#prop = font_manager.FontProperties(fname='/Volumes/Users/shane/Desktop/site_analysis/HelveticaNeueLTStd-Th.otf',)

# custom SITE ANALYSIS libs
from site_analysis import windrose
from site_analysis import temp_rain
from site_analysis import humidity
from site_analysis import psychro
from site_analysis import clouds
from site_analysis import solar

import warnings
warnings.filterwarnings('ignore')

#bom_directory = '/Volumes/Users/shane/Desktop/bom/'
#site = '/Volumes/Users/shane/Desktop/site_analysis/'
#wind_directory = 'wind_data_separated_into_station_number'

sa_folder = sys.argv[1]
bom_directory = 'add_files/bom/'#'/mnt/bom/'
out_path = sa_folder + "/"
wind_directory = 'wind_data_separated_into_station_number'

stns_file = 'add_files/stns/stn_nums_coords.csv'

wind_dir = os.path.join(bom_directory, wind_directory)


def distance_between_coords(stations, coord):
    """
    """
    from geopy.distance import vincenty
    
    return vincenty((stations['lat'], stations['lng']), coord).km

def distance_between_station_and_coord(df, coord):
    """
    """
    df['site_dist'] = df.apply(distance_between_coords, coord=coord, axis=1)
    df['weight'] = 1 / (df['site_dist']**2)
    
    df.sort_values(by=['site_dist'], inplace=True)
    
    return df

def inverse_weight_diff(x, weights):
    """
    """
    return np.sum(weights * x) / np.sum(weights)
    
def main():
    """
    # chatswood
        site = 'Chatswood_Sample'
        lat = -33.79431306
        lng = 151.1810225
        
        address = {'street_no': 1,
                   'street_name': 'Day',
                   'street_type': 'Street',
                   'suburb': 'Chatswood',
                   'state': 'NSW',
                   'postcode': 2152,
                   }
    """
    order_number = sys.argv[1]
    cmap = 'RdYlBu_r'
    #cmap = 'viridis'

    with open('settings.yaml') as f:
        cfg = yaml.safe_load(f)
    for key in cfg:
        globals()[str(key)] = cfg[key]



    #df = get_lats_longs_of_stns()

    df = distance_between_station_and_coord(pd.read_csv(stns_file), (lat, lng))

    from site_analysis import site_analysis_plot

    site_analysis_plot.SiteAnalysisPlot(lat, lng, df, cmap, 
                                        bom_directory, out_path, 
                                        site + '_site_analysis.png',
                                        address
                                        )

if __name__ == '__main__':
    main()
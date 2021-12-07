#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:45:10 2019

@author: shane
"""

import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.interpolate import interp1d
import numpy as np

import seaborn as sns

class WeatherData:
    
    def __init__(self, stn_numbers, bom_dir, weights):
        """
        """
        self.rain_directory = 'rainfall_by_station_number'
        self.high_temp_directory = 'highest_temp_by_station_number'
        self.low_temp_directory = 'lowest_temp_by_station_number'
        self.bom_dir = bom_dir
        
        max_temps = pd.DataFrame()
        min_temps = pd.DataFrame()
        rainfalls = pd.DataFrame()
        
        for stn_number, weight in zip(stn_numbers, weights):
            ###################################################################
            # get the max temperatures
            zip_file = self.get_station_file(stn_number, 
                                             os.path.join(self.bom_dir, 
                                                          self.high_temp_directory)
                                             )
    
            csv_file = self.unzip_temp_file(zip_file)
            
            curr_maxes = self.read_temp_file(csv_file)
            curr_maxes['w'] = np.repeat(weight, curr_maxes.shape[0])
    
            if max_temps.empty:
                max_temps = curr_maxes
            else:
                max_temps = max_temps.append(curr_maxes, ignore_index=True)
        
            ###################################################################
            # gets the minimum temperatures        
            zip_file = self.get_station_file(stn_number, 
                                             os.path.join(self.bom_dir, 
                                                          self.low_temp_directory)
                                             )
    
            csv_file = self.unzip_temp_file(zip_file)

            curr_mins = self.read_temp_file(csv_file)
            curr_mins['w'] = np.repeat(weight, curr_mins.shape[0])
    
            if min_temps.empty:
                min_temps = curr_mins
            else:
                min_temps = min_temps.append(curr_mins, ignore_index=True)
        
            ###################################################################
            # gets rainfall        
            zip_file = self.get_station_file(stn_number, 
                                             os.path.join(self.bom_dir, 
                                                          self.rain_directory)
                                             )
    
            csv_file = self.unzip_temp_file(zip_file)
            
            curr_rain = self.read_rain_file(csv_file)
            curr_rain['w'] = np.repeat(weight, curr_rain.shape[0])
            
            if rainfalls.empty:
                rainfalls = curr_rain
            else:
                rainfalls = rainfalls.append(curr_rain, ignore_index=True)


        self.max_temps_std = max_temps.groupby('Month').Temp.std()
        self.min_temps_std = min_temps.groupby('Month').Temp.std()
        self.rainfall_std = rainfalls.groupby('Month').Rainfall.std()

        # groups by month and finds the sum then divides by the sum of the 
        # weights, the temps have already been multiplied by the weight
        # before the summation.
        max_temps.Temp = max_temps.Temp * max_temps.w
        min_temps.Temp = min_temps.Temp * min_temps.w
        rainfalls.Rainfall = rainfalls.Rainfall * rainfalls.w
        
        self.max_temps = max_temps.groupby('Month').Temp.sum() / \
                         max_temps.groupby('Month').w.sum()
                          
        self.min_temps = min_temps.groupby('Month').Temp.sum() / \
                         min_temps.groupby('Month').w.sum()
        
        self.rainfall = rainfalls.groupby('Month').Rainfall.sum() / \
                        rainfalls.groupby('Month').w.sum() 
        
        
    def get_station_file(self, stn_number, directory):
        """
        """    
        for root, dirs, files in os.walk(directory):
            for file in files:
                if (str(stn_number) in file) and ('.zip' in file):
                    return os.path.join(root, file)
            
        return None

    def unzip_temp_file(self, zip_file):
        """
        """    
        unzip_dir = zip_file.replace('.zip', '') + '/'
    
        os.system('unzip -qq -o "{}" -d "{}"'.format(zip_file, unzip_dir))
        
        return os.path.join(unzip_dir, 
                            os.path.split(zip_file)[-1].replace('.zip',
                                         '_Data1.csv')
                            )
    
    def read_temp_file(self, temp_file):
        """
        """
        df = pd.read_csv(temp_file, 
                         usecols=[2, 3, 4, 5],                      # use year, month, temp cols
                         names=['Year', 'Month', 'Temp', 'Day'],    # names
                         skiprows=1                                 # skips header
                         ).dropna()
        
        df = df[df.Year >= 2000]
            
        return df
    
    def read_rain_file(self, temp_file):
        """
        """
        df = pd.read_csv(temp_file, 
                         usecols=[2, 3, 4, 5],                 # use year, month, temp cols
                         names=['Year', 'Month', 'Rainfall', 'Quality'],   # names
                         skiprows=1                         # skips header
                         ).dropna()
        
        df = df[df.Year >= 2017]
            
        return df
    
    def merge_dfs(self, df1, df2):
        """
        """
        years = np.append(df1['Year'].values,
                          df2['Year'].values)
        
        months = np.append(df1['Month'].values,
                           df2['Month'].values)
        
        days = np.append(df1['Day'].values,
                         df2['Day'].values)
        
        temps = np.append(df1['Temp'].values,
                          df2['Temp'].values)
        
        return pd.DataFrame({'Year': years, 
                             'Month': months,
                             'Day': days,
                             'Temp': temps,})
    
    
class Temp_Rainfall_Plot:
    
    def __init__(self, weather_data, cmap, font_name):
        """
        """
        print('Creating Temperature/Rainfall Plot.')
        plt.cla()
        
        self.cmap = plt.get_cmap(cmap)
        self.fname = font_name
        
        self.max_temps = weather_data.max_temps
        self.max_temps_std = weather_data.max_temps_std
        
        self.min_temps = weather_data.min_temps
        self.min_temps_std = weather_data.min_temps_std
        
        self.rainfall = weather_data.rainfall
        self.rainfall_std = weather_data.rainfall_std
        
#        self.fig = self.plot_temp_rain(file_name,
#                            self.min_temps,
#                            self.min_temps_std,
#                            self.max_temps,
#                            self.max_temps_std,
#                            self.rainfall
#                            )
        
    def temp_plot_ax(self, ax_temp, min_temps, min_std, max_temps, max_std):
        """
        """
        
        ax_temp.set_alpha(0.0)
        
        N = 10
        sigma = 2
        
        curr_max = -100
        curr_min = 100
        
        xnew = np.linspace(min_temps.index.values.min(), 
                           min_temps.index.values.max(),
                           1000) 
        
        for i in range(N * sigma):
            #######################################################################
            # Minimum temps
            #######################################################################
            # means
            f_mean = interp1d(min_temps.index.values,
                              min_temps.values, 
                              kind='cubic')
            # plus 1/N * std
            f_std_upp = interp1d(min_temps.index.values, 
                                 min_temps.values + ((i / N) * min_std.values), 
                                 kind='cubic')
            # minus 1/N * std
            f_std_low = interp1d(min_temps.index.values,
                                 min_temps.values - ((i / N) * min_std.values),
                                 kind='cubic')
    
            y_mean_mins = f_mean(xnew)
            y_std_upp_mins = f_std_upp(xnew)
            y_std_low_mins = f_std_low(xnew)
            
            #######################################################################
            # Maximum temps
            #######################################################################
            # means
            f_mean = interp1d(max_temps.index.values, 
                              max_temps.values, 
                              kind='cubic')
            # plus 1/N * std
            f_std_upp = interp1d(max_temps.index.values, 
                                 max_temps.values + ((i / N) * max_std.values),
                                 kind='cubic')
            # minus 1/N * std
            f_std_low = interp1d(max_temps.index.values, 
                                 max_temps.values - ((i / N) * max_std.values),
                                 kind='cubic')
           
            y_mean_maxs = f_mean(xnew)
            y_std_upp_maxs = f_std_upp(xnew)
            y_std_low_maxs = f_std_low(xnew)
        
            #######################################################################
            # stack together for colormapping
            #######################################################################
            
            x_all = np.vstack((xnew, xnew, xnew, xnew, xnew, xnew))
            
            y_all = np.vstack((y_std_low_mins, y_mean_mins, y_std_upp_mins,
                               y_std_low_maxs, y_mean_maxs, y_std_upp_maxs))
            
            y_max = y_all.max()
            y_min = y_all.min()
            
            if y_max > curr_max:
                curr_max = y_max
                
            if y_min < curr_min:
                curr_min = y_min
                
            ax_temp.scatter(x_all, y_all, c=y_all, cmap=self.cmap, marker='s', 
                            s=1.55, alpha=0.25, zorder=2)
        
        min_lin_color = 'darkgrey'
        max_line_color = 'darkgrey'
        
        back_line = 0.8
        
        ###########################################################################
        # plot the mean and error ranges 
        ###########################################################################
        #ax_temp.plot(xnew, y_std_upp_mins, lw=back_line, color=min_lin_color, ls='-')
        #ax_temp.plot(xnew, y_std_low_mins, lw=back_line, color=min_lin_color, ls='-')
        ax_temp.plot(xnew, y_mean_mins, lw=back_line, color=min_lin_color, ls='-')
#        ax_temp.plot(xnew, y_std_upp_mins, lw=fore_line, color=mid_line, ls='--')
#        ax_temp.plot(xnew, y_std_low_mins, lw=fore_line, color=mid_line, ls='--')
#        ax_temp.plot(xnew, y_mean_mins, lw=fore_line, color=mid_line, ls='--')
        
        #ax_temp.plot(xnew, y_std_upp_maxs, lw=back_line, color=max_line_color, ls='-')
        #ax_temp.plot(xnew, y_std_low_maxs, lw=back_line, color=max_line_color, ls='-')
        ax_temp.plot(xnew, y_mean_maxs, lw=back_line, color=max_line_color, ls='-')
#        ax_temp.plot(xnew, y_std_upp_maxs, lw=fore_line, color=mid_line, ls='--')
#        ax_temp.plot(xnew, y_std_low_maxs, lw=fore_line, color=mid_line, ls='--')
#        ax_temp.plot(xnew, y_mean_maxs, lw=fore_line, color=mid_line, ls='--')   
        
        ###########################################################################
            # stack together for colormapping
        ###########################################################################
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=10)
                
        ax_temp.set_facecolor('white')
        ax_temp.set_ylabel(r'Max/Min Temperature [$^{\circ}$C]', fontproperties=prop)
        ax_temp.yaxis.set_label_coords(-0.05,0.5)
        ax_temp.set_yticks([0, 10, 20, 30, 40, 50])
        ax_temp.set_yticklabels([0, 10, 20, 30, 40], fontproperties=prop)
        ax_temp.set_xlabel('', fontproperties=prop)

        ax_temp.set_xticks([x + 1 for x in range(12)])
        ax_temp.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
        fontproperties=prop)
        
        ax_temp.grid(zorder=1)
        ax_temp.tick_params(pad=1.0, grid_lw=0.2)
        
        return ax_temp
    
    def rain_plot_ax(self, ax_rain, rainfall):
        """
        """
        sns.barplot(x=rainfall.index,
                    y=rainfall.values,
                    color=self.cmap(0.5),
                    edgecolor='darkgrey',
                    ax=ax_rain,
                    ci=None,
                    zorder=2,
                    alpha=0.7) 
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=10)
        
        ax_rain.set_xlabel('Month', fontproperties=prop)
        
        ax_rain.set_facecolor('white')
        ax_rain.set_ylabel('Average Monthly Rainfall [mm]', fontproperties=prop)
        ax_rain.yaxis.set_label_coords(-0.05,0.5)
        ax_rain.set_yticklabels(np.arange(0, 1000, 50), fontproperties=prop)
        ax_rain.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontproperties=prop)
        
        ax_rain.grid(zorder=1)
        ax_rain.tick_params(pad=1.0, grid_lw=0.2)
        
        return ax_rain
    
    def table_plot_ax(self, fig, min_temps, min_std, max_temps, max_std, 
                       rainfall):
        """
        """
        sigma = 2
        ax_table = fig.add_axes([0.56, 0.43, 0.42, 0.13], zorder=2)
        ax_table.axis([-0.5, 11.5, 0.0, 1.0])
        ax_table.axis('off')
    
        x_pos = np.arange(-0.5, 11.5, 1) + 0.07
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                  'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
        hatch_locs = np.digitize(rainfall.round(), 
                                 np.array([0., 25., 50., 75., 100., 150., 200., np.inf]),
                                 right=True)
        
        rect_background = 'white'
        rect_width = 0.9
    
        rect_height = 0.15
        
        # height at which boxes start
        y_height = 0.55
        
        hatches = ['/' * x for x in hatch_locs]
        
        low_limit = min_temps - sigma * min_std
        high_limit = max_temps + sigma * max_std
        
        t_norm = mpl.colors.Normalize(vmin=low_limit.min(), 
                                      vmax=high_limit.max())
        
        box_x_pos = 0.5625
        
        imshow_width = 0.0315
        imshow_height = 0.021
        imshow_x_shift = 0.035

        for r in range(12):
            ax_test = fig.add_axes([box_x_pos + (imshow_x_shift * r), 0.482, imshow_width, imshow_height], zorder=1)
            ax_test.axis([0, 1, 0, 1])
    
            low = max_temps.values[r] - (sigma * max_std.values[r])
            high = max_temps.values[r] + (sigma * max_std.values[r])
            
            ax_test.imshow([[low, low], [high, high]], 
                           cmap = self.cmap, 
                           norm=t_norm,
                           interpolation = 'bicubic',
                           aspect='auto',
                           zorder=1,
                           alpha=0.8
                           )
            
            ax_test.axis('off')
            
            ax_test = fig.add_axes([box_x_pos + (imshow_x_shift * r), 0.4615, imshow_width, imshow_height], zorder=1)
            ax_test.axis([0, 1, 0, 1])
    
            low = min_temps.values[r] - (sigma * min_std.values[r])
            high = min_temps.values[r] + (sigma * min_std.values[r])
            
            ax_test.imshow([[low, low], [high, high]], 
                           cmap = self.cmap, 
                           norm=t_norm,
                           interpolation = 'bicubic',
                           aspect='auto',
                           zorder=1,
                           alpha=0.8
                           )
            
            ax_test.axis('off')
        
        ###########################################################################
        # rectangles for the table
        ###########################################################################
        for n, x in enumerate(x_pos):
            rect = plt.Rectangle((x, y_height), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor=rect_background, edgecolor='black')
            ax_table.add_patch(rect)
            
            rect = plt.Rectangle((x, y_height - 0.15), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor='none', edgecolor='black', alpha=0.7)
            ax_table.add_patch(rect)
    
            rect = plt.Rectangle((x, y_height - (2 * 0.15)), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor='none', edgecolor='black', alpha=0.75)
            ax_table.add_patch(rect)
            
            rect = plt.Rectangle((x, y_height - (3 * 0.15)), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor='none', edgecolor='blue', hatch=hatches[n], alpha=0.2)
            ax_table.add_patch(rect)
            
            rect = plt.Rectangle((x, y_height - (3 * 0.15)), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor='none', edgecolor='black')
            ax_table.add_patch(rect)
            
        height = 0.595
        font = 12
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=font)
        
        ax_table.text(-1.9, height - 0.15, r'Max [$^{\circ}$C]', 
                      fontproperties=prop)
        ax_table.text(-1.9, height - (2 * 0.15), r'Min [$^{\circ}$C]', 
                      fontproperties=prop)
        ax_table.text(-1.9, height - (3 * 0.15), 'Rain [mm]',
                      fontproperties=prop)
        
        back_text_color = 'black'
     
        ###########################################################################
        # plot values in the table 
        ###########################################################################
        for x in np.arange(0, 12, 1):
            # month
            ax_table.text(x, height, months[x], fontproperties=prop, ha='center',
                          zorder=3)
            # max
            ax_table.text(x, 
                          height - 0.175, 
                          '{:.1F}'.format(max_temps.values[x]), 
                          color=back_text_color, 
                          ha='center',
                          va='bottom',
                          zorder=3, 
                          fontproperties=prop)
             # min
            ax_table.text(x, 
                          height - (2 * 0.162), 
                          '{:.1F}'.format(min_temps.values[x]), 
                          color=back_text_color, 
                          ha='center',
                          va='bottom',
                          zorder=3, 
                          fontproperties=prop)
            # rainfall
            ax_table.text(x, 
                          height - (3 * 0.16), 
                          '{:.0F}'.format(rainfall.values[x]), 
                          color=back_text_color, 
                          ha='center',
                          va='bottom',
                          zorder=3, 
                          fontproperties=prop)
            #txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground=fore_text_color)])
    
        
    def plot_temp_rain(self, file_name, min_temps, min_std, max_temps, max_std, 
                       rainfall):
        """
        """                
        fig, ax = plt.subplots(figsize=(12, 6))
        
        fig.patch.set_alpha(0.0)
        
        ax.axis([-10, 10, -10, 10])
        
        ax.axis('off')
        
        ax_temp = fig.add_axes([0.13, 0.7, 0.8, 0.25])
        
        ax_temp.set_alpha(0.0)
        
        N = 10
        sigma = 2
        
        curr_max = -100
        curr_min = 100
        
        xnew = np.linspace(min_temps.index.values.min(), 
                           min_temps.index.values.max(),
                           1000) 
        
        for i in range(N * sigma):
            #######################################################################
            # Minimum temps
            #######################################################################
            # means
            f_mean = interp1d(min_temps.index.values,
                              min_temps.values, 
                              kind='cubic')
            # plus 1/N * std
            f_std_upp = interp1d(min_temps.index.values, 
                                 min_temps.values + ((i / N) * min_std.values), 
                                 kind='cubic')
            # minus 1/N * std
            f_std_low = interp1d(min_temps.index.values,
                                 min_temps.values - ((i / N) * min_std.values),
                                 kind='cubic')
    
            y_mean_mins = f_mean(xnew)
            y_std_upp_mins = f_std_upp(xnew)
            y_std_low_mins = f_std_low(xnew)
            
            #######################################################################
            # Maximum temps
            #######################################################################
            # means
            f_mean = interp1d(max_temps.index.values, 
                              max_temps.values, 
                              kind='cubic')
            # plus 1/N * std
            f_std_upp = interp1d(max_temps.index.values, 
                                 max_temps.values + ((i / N) * max_std.values),
                                 kind='cubic')
            # minus 1/N * std
            f_std_low = interp1d(max_temps.index.values, 
                                 max_temps.values - ((i / N) * max_std.values),
                                 kind='cubic')
           
            y_mean_maxs = f_mean(xnew)
            y_std_upp_maxs = f_std_upp(xnew)
            y_std_low_maxs = f_std_low(xnew)
        
            #######################################################################
            # stack together for colormapping
            #######################################################################
            
            x_all = np.vstack((xnew, xnew, xnew, xnew, xnew, xnew))
            
            y_all = np.vstack((y_std_low_mins, y_mean_mins, y_std_upp_mins,
                               y_std_low_maxs, y_mean_maxs, y_std_upp_maxs))
            
            y_max = y_all.max()
            y_min = y_all.min()
            
            if y_max > curr_max:
                curr_max = y_max
                
            if y_min < curr_min:
                curr_min = y_min
                
            ax_temp.scatter(x_all, y_all, c=y_all, cmap=self.cmap, marker='s', s=1.55, alpha=0.25, zorder=2)
        
        min_lin_color = 'darkgrey'
        max_line_color = 'darkgrey'
        
        back_line = 0.8
        
        ###########################################################################
        # plot the mean and error ranges 
        ###########################################################################
        #ax_temp.plot(xnew, y_std_upp_mins, lw=back_line, color=min_lin_color, ls='-')
        #ax_temp.plot(xnew, y_std_low_mins, lw=back_line, color=min_lin_color, ls='-')
        ax_temp.plot(xnew, y_mean_mins, lw=back_line, color=min_lin_color, ls='-')
#        ax_temp.plot(xnew, y_std_upp_mins, lw=fore_line, color=mid_line, ls='--')
#        ax_temp.plot(xnew, y_std_low_mins, lw=fore_line, color=mid_line, ls='--')
#        ax_temp.plot(xnew, y_mean_mins, lw=fore_line, color=mid_line, ls='--')
        
        #ax_temp.plot(xnew, y_std_upp_maxs, lw=back_line, color=max_line_color, ls='-')
        #ax_temp.plot(xnew, y_std_low_maxs, lw=back_line, color=max_line_color, ls='-')
        ax_temp.plot(xnew, y_mean_maxs, lw=back_line, color=max_line_color, ls='-')
#        ax_temp.plot(xnew, y_std_upp_maxs, lw=fore_line, color=mid_line, ls='--')
#        ax_temp.plot(xnew, y_std_low_maxs, lw=fore_line, color=mid_line, ls='--')
#        ax_temp.plot(xnew, y_mean_maxs, lw=fore_line, color=mid_line, ls='--')   
        
        ###########################################################################
            # stack together for colormapping
        ###########################################################################
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=12)
        ax_temp.set_facecolor('white')
        ax_temp.set_ylabel(r'Temperature [$^{\circ}$C]', fontproperties=prop)
        ax_temp.set_yticks([0, 10, 20, 30, 40, 50])
        ax_temp.set_yticklabels([0, 10, 20, 30, 40], fontproperties=prop)
        ax_temp.set_xlabel('', fontproperties=prop)

        ax_temp.set_xticks([x + 1 for x in range(12)])
        ax_temp.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontproperties=prop)
        
        ax_temp.grid(zorder=1)
        
    ###############################################################################
    # rainfall bar graph/histogram
    ###############################################################################
        
        ax_rain = fig.add_axes([0.13, 0.37, 0.8, 0.25])
    
        sns.barplot(x=rainfall.index,
                    y=rainfall.values,
                    color=self.cmap(0.5),
                    edgecolor='darkgrey',
                    ax=ax_rain,
                    ci=None,
                    zorder=2,
                    alpha=0.7) 
        
        ax_rain.set_xlabel('Month', fontproperties=prop)
        
        ax_rain.set_facecolor('white')
        ax_rain.set_ylabel('Average Monthly Rainfall [mm]', fontproperties=prop)
        ax_rain.set_yticklabels(np.arange(0, 1000, 50), fontproperties=prop)
        ax_rain.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontproperties=prop)
        
        ax_rain.grid(zorder=1)
        
    ###############################################################################
    # table axes
    ###############################################################################    
        
        ax_table = fig.add_axes([0.13, 0.02, 0.8, 0.35], zorder=2)
        ax_table.axis([*ax_rain.get_xlim(), 0.0, 1.0])
        ax_table.axis('off')
    
        x_pos = np.arange(*ax_rain.get_xlim(), 1) + 0.07
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                  'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
        hatch_locs = np.digitize(rainfall.round(), 
                                 np.array([0., 25., 50., 75., 100., 150., 200., np.inf]),
                                 right=True)
        
        rect_background = 'white'
        rect_width = 0.9
    
        rect_height = 0.15
        
        # height at which boxes start
        y_height = 0.55
        
        hatches = ['/' * x for x in hatch_locs]
        
        low_limit = min_temps - sigma * min_std
        high_limit = max_temps + sigma * max_std
        
        t_norm = mpl.colors.Normalize(vmin=low_limit.min(), 
                                      vmax=high_limit.max())
        
        box_x_pos = 0.135

        for r in range(12):
            ax_test = fig.add_axes([box_x_pos + (0.06665 * r), 0.1611, 0.060, 0.050], zorder=1)
            ax_test.axis([0, 1, 0, 1])
    
            low = max_temps.values[r] - (sigma * max_std.values[r])
            high = max_temps.values[r] + (sigma * max_std.values[r])
            
            ax_test.imshow([[low, low], [high, high]], 
                           cmap = self.cmap, 
                           norm=t_norm,
                           interpolation = 'bicubic',
                           aspect='auto',
                           zorder=1,
                           alpha=0.8
                           )
            
            ax_test.axis('off')
            
            ax_test = fig.add_axes([box_x_pos + (0.06665 * r), 0.108, 0.060, 0.050], zorder=1)
            ax_test.axis([0, 1, 0, 1])
    
            low = min_temps.values[r] - (sigma * min_std.values[r])
            high = min_temps.values[r] + (sigma * min_std.values[r])
            
            ax_test.imshow([[low, low], [high, high]], 
                           cmap = self.cmap, 
                           norm=t_norm,
                           interpolation = 'bicubic',
                           aspect='auto',
                           zorder=1,
                           alpha=0.8
                           )
            
            ax_test.axis('off')
        
        ###########################################################################
        # rectangles for the table
        ###########################################################################
        for n, x in enumerate(x_pos):
            rect = plt.Rectangle((x, y_height), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor=rect_background, edgecolor='black')
            ax_table.add_patch(rect)
            
            rect = plt.Rectangle((x, y_height - 0.15), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor='none', edgecolor='black', alpha=0.7)
            ax_table.add_patch(rect)
    
            rect = plt.Rectangle((x, y_height - (2 * 0.15)), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor='none', edgecolor='black', alpha=0.75)
            ax_table.add_patch(rect)
            
            rect = plt.Rectangle((x, y_height - (3 * 0.15)), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor='none', edgecolor='blue', hatch=hatches[n], alpha=0.2)
            ax_table.add_patch(rect)
            
            rect = plt.Rectangle((x, y_height - (3 * 0.15)), width=rect_width, height=rect_height, zorder=3,
                                 fill=True, facecolor='none', edgecolor='black')
            ax_table.add_patch(rect)
            
        height = 0.6
        font = 16
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=font)
        
        ax_table.text(-1.9, height - 0.15, r'Max [$^{\circ}$C]', 
                      fontproperties=prop)
        ax_table.text(-1.9, height - (2 * 0.15), r'Min [$^{\circ}$C]', 
                      fontproperties=prop)
        ax_table.text(-1.9, height - (3 * 0.15), 'Rain [mm]',
                      fontproperties=prop)
        
        back_text_color = 'black'
     
        ###########################################################################
        # plot values in the table 
        ###########################################################################
        for x in np.arange(0, 12, 1):
            # month
            ax_table.text(x, height, months[x], fontproperties=prop, ha='center',
                          zorder=3)
            # max
            ax_table.text(x, 
                          height - 0.17, 
                          '{:.1F}'.format(max_temps.values[x]), 
                          color=back_text_color, 
                          ha='center',
                          va='bottom',
                          zorder=3, 
                          fontproperties=prop)
             # min
            ax_table.text(x, 
                          height - (2 * 0.16), 
                          '{:.1F}'.format(min_temps.values[x]), 
                          color=back_text_color, 
                          ha='center',
                          va='bottom',
                          zorder=3, 
                          fontproperties=prop)
            # rainfall
            ax_table.text(x, 
                          height - (3 * 0.16), 
                          '{:.0F}'.format(rainfall.values[x]), 
                          color=back_text_color, 
                          ha='center',
                          va='bottom',
                          zorder=3, 
                          fontproperties=prop)
            #txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground=fore_text_color)])
    
        ###########################################################################
        # align the x and y axis labels 
        ###########################################################################
        ax_temp.yaxis.set_label_coords(-0.075, 0.5)
        ax_rain.yaxis.set_label_coords(-0.075, 0.5)
        
        ax_temp.xaxis.set_label_coords(0.5, -0.25)
        ax_rain.xaxis.set_label_coords(0.5, -0.25)
    
        fig.savefig(file_name, dpi=240)
        
        return fig
        
"""
stn_number = 66062

plot_temp_rain(file_name,
               mins.groupby('Month').mean(),
               mins.groupby('Month').std(),
               maxes.groupby('Month').mean(),
               maxes.groupby('Month').std(),
               rainfall.groupby('Month').mean())
"""
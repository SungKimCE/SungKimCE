#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:02:55 2019

@author: shane
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


angles = {'N': 90.0,
          'NNE': 77.5,
          'NE': 45.0,
          'ENE': 22.5,
          'E': 0,
          'ESE': 337.5,
          'SE': 315.0,
          'SSE': 292.5,
          'S': 270.0,
          'SSW': 247.5,
          'SW': 225.0,
          'WSW': 202.5,
          'W': 180.0,
          'WNW': 157.5,
          'NW': 135.0,
          'NNW': 112.5,
          }

seasons = {'Summer': [12, 1, 2],
           'Autumn': [3, 4, 5],
           'Winter': [6, 7, 8],
           'Spring': [9, 10, 11]
           }

# this determines the increments to round the max seasonal percentage up to,
# i.e. base = 5.0, then max percentage is rounded up to the nearest 5%
base = 1.0

cardinal_order = ['N', 'NNW', 'NW', 'WNW', 'W', 'WSW', 'SW', 'SSW', 
                  'S', 'SSE', 'SE', 'ESE', 'E', 'ENE', 'NE', 'NNE']

    
################################################################################
    # windrose data class
class WindData:
    
    def __init__(self, stn_numbers, wind_dir, weights, num_speed_bins=9):
        """
            Reads in the wind data from the wind data file and bins the data 
            and separates the data into speed, gust and seasonal. The data
            is assigned as attributes of the object.
            
            The attributes are: 
                    
                wind_rose_data, wind_gust_data, Summer, Autumn,
                Winter, Spring, seasonal_spd_bins, max_seasonal_pc,
                speed_stats
            
            Parameters
            ----------
                    stn_numbers : float
                    The weather station numbers to get the wind data for.
                    
                    wind_dir : string
                    Location of the wind data files.
                    
                    weights : list, ndarray
                    The weighting for each stn.
                    
                    num_speed_bins : int, optional
                    Number of bins to put the wind speed data into.
                    
            Returns
            -------
                    None
        """
        self.stn_numbers = stn_numbers
        self.wind_dir = wind_dir
        self.num_speed_bins = num_speed_bins
        
        all_wind = []
        all_gust = []
        
        all_summer = []
        all_autumn = []
        all_spring = []
        all_winter = []
        
        for stn in self.stn_numbers:
            ###################################################################
            # wind data, the spd_bins returned here are the bins that will be
            # used for the seasonal windrose
            wind_file = os.path.join(self.wind_dir,
                                     'wind_data_stn_{}.csv'.format(stn)
                                     )
            
            all_wind.append(self.get_wind(wind_file))
            all_gust.append(self.get_wind_gust(wind_file))
            
            # summer data
            all_summer.append(self.get_seasonal_wind(wind_file, seasons['Summer']))
            
            # autumn data
            all_autumn.append(self.get_seasonal_wind(wind_file, seasons['Autumn']))
            
            # spring data
            all_spring.append(self.get_seasonal_wind(wind_file, seasons['Spring']))
            
            # winter data
            all_winter.append(self.get_seasonal_wind(wind_file, seasons['Winter']))

        self.spd_bins, self.speed_stats = self.make_spd_bins(all_wind, num_speed_bins)
        
        self.spd_labels = self.speed_labels(self.spd_bins)
        
        spd_bins, self.gust_speed_stats = self.make_spd_bins(all_gust, num_speed_bins)
        
        spd_labels = self.speed_labels(spd_bins)
        
        wind_rose_data = self.make_binned_frame(all_wind, self.spd_bins, self.spd_labels, weights)
        wind_gust_data = self.make_binned_frame(all_gust, spd_bins, spd_labels, weights)
        
        self.wind_rose_data = wind_rose_data.loc[:, (wind_rose_data != 0).any(axis=0)]
        self.wind_gust_data = wind_gust_data.loc[:, (wind_gust_data != 0).any(axis=0)]
        
        summer = self.make_binned_frame(all_summer, self.spd_bins, self.spd_labels, weights)
        self.Summer = summer.loc[:, (summer != 0).any(axis=0)]
        
        autumn = self.make_binned_frame(all_autumn, self.spd_bins, self.spd_labels, weights)
        self.Autumn = autumn.loc[:, (autumn != 0).any(axis=0)]
        
        spring = self.make_binned_frame(all_spring, self.spd_bins, self.spd_labels, weights)
        self.Spring = spring.loc[:, (spring != 0).any(axis=0)]
        
        winter = self.make_binned_frame(all_winter, self.spd_bins, self.spd_labels, weights)
        self.Winter = winter.loc[:, (winter != 0).any(axis=0)]

        max_seasonal_pc = np.max([self.Summer.sum(axis=1).max(),
                                  self.Autumn.sum(axis=1).max(),
                                  self.Spring.sum(axis=1).max(),
                                  self.Winter.sum(axis=1).max()])
    
        self.max_seasonal_pc = base * np.ceil(max_seasonal_pc / base)
            
        self.speed_stats = self.speed_stats + self.gust_speed_stats
        
        self.wind = all_wind
    
    def update_wind_frame(self, old_frame, new_frame, weights):
        """
        """            
        new_frame['weight'] = np.repeat(weights, new_frame.shape[0])
            
        if old_frame.empty and not new_frame.empty:
            old_frame = new_frame
        else:
            old_frame = old_frame.append(new_frame, ignore_index=True)
            
        return old_frame
        
    def get_wind(self, wind_file):
        """
        """
        df = pd.read_csv(wind_file, dtype=str)
        
        df_9am = df[['9am_wind_dir', 
                     '9am_wind_speed']].dropna()
        
        df_9am.rename(index=str, 
                      columns={"9am_wind_dir": "wind_dir",
                               "9am_wind_speed": "wind_speed"},
                               inplace=True
                      )
        
        df_3pm = df[['3pm_wind_dir',
                     '3pm_wind_speed']].dropna()
        
        df_3pm.rename(index=str, 
                      columns={"3pm_wind_dir": "wind_dir",
                               "3pm_wind_speed": "wind_speed"},
                               inplace=True
                      )
        
        return self.merge_wind_dfs(df_9am, df_3pm)
            
    def merge_wind_dfs(self, df1, df2):
        """
        """
        wind_dirs = np.append(df1['wind_dir'].values,
                              df2['wind_dir'].values)
        
        wind_speeds = np.append(df1['wind_speed'].values,
                                df2['wind_speed'].values)
        
        return pd.DataFrame({'wind_dirs': wind_dirs, 
                             'wind_speeds': wind_speeds}) 
            
    def get_seasonal_wind(self, wind_file, season_months):
        """
        """
        df = pd.read_csv(wind_file, dtype=str)
    
        df['month'] = pd.DatetimeIndex(df['Date']).month
    
        mask = df['month'].isin(season_months)
        
        df = df[mask]
        
        df_9am = df[['9am_wind_dir', 
                     '9am_wind_speed']].dropna()
        
        df_9am.rename(index=str, 
                      columns={"9am_wind_dir": "wind_dir",
                               "9am_wind_speed": "wind_speed"},
                               inplace=True
                      )
        
        df_3pm = df[['3pm_wind_dir',
                     '3pm_wind_speed']].dropna()
        
        df_3pm.rename(index=str, 
                      columns={"3pm_wind_dir": "wind_dir",
                               "3pm_wind_speed": "wind_speed"},
                               inplace=True
                      )
        
        return self.merge_wind_dfs(df_9am, df_3pm)
    
    def get_wind_gust(self, wind_file):
        """
        """
        df = pd.read_csv(wind_file, dtype=str)
        
        df_wind_gust = df[['direction_max_wind_gust', 
                           'speed_max_wind_gust']].dropna()
    
        df_wind_gust.rename(index=str, 
                            columns={"direction_max_wind_gust": "wind_dirs",
                                     "speed_max_wind_gust": "wind_speeds"},
                            inplace=True
                      )
        
        return df_wind_gust
    
    def make_spd_bins(self, wind_data, num_speed_bins):
        """
        """
        max_speed = -99
        min_speed = 1e5
        
        total_wind = 0
        wind_numbers = 0
        
        for wind in wind_data:
            total_wind += wind.wind_speeds[wind.wind_speeds != 'Calm'].astype('float').sum()
            wind_numbers += wind.wind_speeds[wind.wind_speeds != 'Calm'].astype('float').count()
            
            curr_min = wind.wind_speeds[wind.wind_speeds != 'Calm'].astype('float').min()
            curr_max = wind.wind_speeds[wind.wind_speeds != 'Calm'].astype('float').max()
            
            if curr_min < min_speed:
                min_speed = curr_min

            if curr_max > max_speed:
                max_speed = curr_max
                
        return [-np.inf] + \
        list(np.round(np.logspace(np.log10(min_speed), 
                                  np.log10(max_speed),
                                  num_speed_bins))) + \
                                  [np.inf], (max_speed, total_wind / wind_numbers)
        
    def make_binned_frame(self, wind_frame, spd_bins, spd_labels, weights):
        """
        """
        binned_frame = []
        
        for frame in wind_frame:
            binned_frame.append(self.bin_wind_data(frame, spd_bins, spd_labels))
        
        weighted_binned_frame = (((weights[0] * binned_frame[0]) + \
                                  (weights[1] * binned_frame[1]) + \
                                  (weights[2] * binned_frame[2])) / \
                                weights.sum())
                         
        return weighted_binned_frame / weighted_binned_frame.sum(axis=1).sum() * 100.
    
    def bin_wind_data(self, wind_data, spd_bins, spd_labels):
        """
        """
        mask = wind_data['wind_speeds'] == 'Calm'
        
        calm_count = wind_data['wind_speeds'][mask].count()
            
        wind_data = wind_data[-mask]
        wind_data.loc[:,'wind_speeds'] = wind_data['wind_speeds'].astype('float')
        
        rose = wind_data.assign(WindSpd_bins=lambda df:
                pd.cut(wind_data['wind_speeds'], bins=spd_bins, labels=spd_labels, right=True)
                ).assign(WindDir_bins=wind_data['wind_dirs']).groupby(by=['WindSpd_bins', 'WindDir_bins']
                ).size(
                        ).unstack(level='WindSpd_bins'
                        ).fillna(0
                        ).assign(calm=lambda wind_data: calm_count / wind_data.shape[0]
                        )
            
        # adds directions not in rose frame, fills with zeros
        if rose.shape[0] != 16:
            dirs = list(angles.keys())
            have_dirs = list(rose.index)
    
            diff = list(np.setdiff1d(dirs, have_dirs))
    
            rose = rose.reindex(rose.index.values.tolist() + diff)
            rose = rose.sort_index().fillna(0)
            
        rose = rose.reindex(spd_labels, axis=1)
        # resets the calm number, used for when there are directional 
        # rows missing
        rose['calm'] = calm_count / 16

        for col in rose.columns:
            if rose[col].isna().all():
                rose[col] = 0
                
        return rose.reindex(cardinal_order).reindex(spd_labels, axis=1)
    
    def round_pc(self, max_pc_frame):
        """
        """
        max_rounded = np.round(max_pc_frame / 2.0) * 2.0
        
        if max_pc_frame > max_rounded:
            return max_rounded + 2
        
        return max_rounded
    
    def speed_labels(self, bins): 
        """
        """
        labels = []
        
        for left, right in zip(bins[:-1], bins[1:]):
            if left == bins[0]:
                labels.append('calm'.format(right))
            elif np.isinf(right):
                labels.append('>{:.0F}'.format(left))
            else:
                labels.append('{:.0F} - {:.0F}'.format(left, right))
    
        return list(labels) 
    
################################################################################
    # windrose plot class
class WindRosePlot:
    
    def __init__(self, wind_package, cmap, font_name):
        """
            Creates windrose diagrams for wind speeeds, wind gusts and
            seasonal wind speed rose diagrams from the wind_package,
            which is a WindData object.
            
            Parameters
            ----------
                    out_dir : string
                    Path where the windrose diagrams are saved.
                    
                    out_file_name : string
                    The output file name without a file extension.
                    
                    wind_package : WindData object
                    Stores the wind rose, gust and seasonal data
                    formatted in the correct pandas dataframe.
                    
            Returns
            -------
                    None
        """
        plt.cla()
        
        self.cmap = plt.get_cmap(cmap)
        self.fname = font_name
        
        self.wind_rose_data = wind_package.wind_rose_data
        self.wind_gust_data = wind_package.wind_gust_data
        self.summer = wind_package.Summer
        self.autumn = wind_package.Autumn
        self.spring = wind_package.Spring
        self.winter = wind_package.Winter
        self.seasonal_spd_bins = self.summer.columns
        self.seasonal_max_pc = wind_package.max_seasonal_pc
        self.speed_stats = wind_package.speed_stats
        
        color_vals, vals = self.wind_rose(self.wind_rose_data, 
                                          self.speed_stats,)
        
        self.wind_rose(self.wind_gust_data, 
                       self.speed_stats, gust=True)
        
        self.seasonal_windrose(self.summer, 
                               self.autumn, self.spring, self.winter, 
                               self.seasonal_spd_bins, self.seasonal_max_pc,
                               color_vals, vals)
        
    def wind_rose_ax(self, rosedata, ax_rose, ax_color, speeds=None, 
                     gust=None, seasonal=None, seasonal_max_pc=None):
        """
        """
        bar_dir, bar_width = self.wind_rose_plot_dirs()
        
        ax_rose.set_theta_direction('counterclockwise')
        ax_rose.set_theta_zero_location('N')
        ax_rose.set_rlabel_position(30)
        
        if seasonal_max_pc:
            fontsize = 8
            padding = -4.0
        else:
            fontsize = 12
            padding = 0.0
        
        color_vals = []
    
        for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
            
            if n == 0:
                color_val = float((n) / (rosedata.shape[1] - 1))
                color_vals.append(color_val)
                
                ax_rose.bar(bar_dir, rosedata[c1].values, 
                            width=bar_width,
                            edgecolor='none',
                            color=self.cmap(color_val),
                            linewidth=0,)

            color_val = float((n + 1) / (rosedata.shape[1] - 1))
            color_vals.append(color_val)
            
            ax_rose.bar(bar_dir, rosedata[c2].values, 
                        width=bar_width, 
                        bottom=rosedata.cumsum(axis=1)[c1].values,
                        color=self.cmap(color_val),
                        edgecolor='none',
                        linewidth=0,
                        )
                
        cardinals = ['N', '', 'NW', '', 'W', '', 'SW', '', 
                     'S', '', 'SE', '', 'E', '', 'NE', '']
        percentages = ['{}%'.format(x) for x in range(100)]
        
        prop = font_manager.FontProperties(fname=self.fname, size=fontsize)
            
        ax_rose.set_rgrids([1, 2, 3, 4, 5, 6])
        ax_rose.tick_params(pad=padding, grid_lw=0.2)
        ax_rose.set_xticks(np.linspace(0, 2 * np.pi, 17))
        ax_rose.set_xticklabels(cardinals,
                                fontproperties=prop,
                                zorder=7)
        if seasonal_max_pc:
            max_pc = seasonal_max_pc
        else:        
            max_pc = rosedata.sum(axis=1).max()
            
        print('max_pc', max_pc, 'seasonal_max', seasonal_max_pc)
        
        if max_pc > 10:
            ax_rose.set_yticks(np.arange(0, 100, 4))
            ax_rose.set_yticklabels(percentages[0::4], 
                                fontproperties=prop,
                                zorder=8)
        elif max_pc > 25:
            ax_rose.set_yticks(np.arange(0, 100, 5))
            ax_rose.set_yticklabels(percentages[0::5], 
                                fontproperties=prop,
                                zorder=8)  
        else:
            ax_rose.set_yticks(np.arange(0, 100, 2))
            ax_rose.set_yticklabels(percentages[0::2], 
                                fontproperties=prop,
                                zorder=8)    
            
        if seasonal_max_pc:
            ax_rose.set_ylim(0, seasonal_max_pc)
        else:        
            ax_rose.set_ylim(0, 1.0 * np.ceil(rosedata.sum(axis=1).max() / 1.0))
            
        if ax_color:
            ax_color = self.color_bar_ax(list(rosedata.columns),
                                         rosedata,
                                         ax_color,
                                         gust,
                                         seasonal,
                                         color_vals)
            fontsize = 10
            
            prop = font_manager.FontProperties(fname=self.fname, size=fontsize)
            
            if not gust:
                ax_color.text(-6.5, -0.12, 
                                 'Mean: {:.1F} km/h, Max: {:.1F} km/h'.format(self.speed_stats[1], self.speed_stats[0]), ha='left',
                                 fontproperties=prop)
#                ax_color.text(7.7, -0.12, 
#                                 '{:.1F} km/h'.format(self.speed_stats[1]), ha='right',
#                                 fontproperties=self.prop, fontsize=fontsize)
            
#                ax_color.text(-6, -0.19, 
#                                 'Max: {:.1F} km/h'.format(self.speed_stats[0]), ha='left',
#                                 fontproperties=self.prop, fontsize=fontsize)
#                ax_color.text(7.7, -0.19, 
#                                 '{:.1F} km/h'.format(self.speed_stats[0]), ha='right',
#                                 fontproperties=self.prop, fontsize=fontsize)
            elif gust and not seasonal_max_pc:
                ax_color.text(-6.5, -0.12, 
                                 'Mean: {:.1F} km/h, Max: {:.1F} km/h'.format(self.speed_stats[3], self.speed_stats[2]), ha='left',
                                 fontproperties=prop)
#                ax_color.text(7.7, -0.12, 
#                                 '{:.1F} km/h'.format(self.speed_stats[1]), ha='right',
#                                 fontproperties=self.prop, fontsize=fontsize)
            
#                ax_color.text(-3, -0.19, 
#                                 'Max: {:.1F} km/h'.format(self.speed_stats[0]), ha='left',
#                                 fontproperties=self.prop, fontsize=fontsize)
        return ax_rose, ax_color
        
    def color_bar_ax(self, spd_labels, rosedata, ax, gust, seasonal, color_vals):
        """
        """
        y_pos = 0.025
        fontsize = 6
        
        prop = font_manager.FontProperties(fname=self.fname, size=fontsize)
        
        labs = rosedata.columns.values
        vals = rosedata.sum(axis=0).values       

        for y_height, color_val, lab in zip(vals / 105, color_vals, labs):
            rect = plt.Rectangle((0.05, y_pos), width=0.9, height=y_height, zorder=2,
                                 fill=True, facecolor=self.cmap(color_val), edgecolor='darkgrey',
                                lw=0.01)
    
            if ('>' not in lab):
                corr = 0.0
                
                if (lab == 'calm') and (y_height * 105 < 2.0):
                    corr = 0.03
                    
                if (lab != 'calm') and (y_height * 105 < 2.0) and (color_val != 1.0):
                    corr = 0.01
                    
                if not seasonal:
                    ax.text(1.05, 
                            (y_pos + 0.5 * y_height) - corr, 
                            lab.title() + ' ({:.1F}%)'.format(y_height * 105), 
                            va='center', 
                            fontproperties=prop)
                else:
                    ax.text(1.05, 
                            (y_pos + 0.5 * y_height) - corr, 
                            lab.title(), 
                            va='center', 
                            fontproperties=prop)
                
            y_pos += y_height
            
            ax.add_patch(rect)
            
        rect = plt.Rectangle((0.05, 0.025), width=0.9, height=y_pos - 0.025, zorder=3,
                             fill=True, facecolor='none', edgecolor='darkgrey', alpha=1.0,
                             lw=1.0)    
        
        ax.add_patch(rect)
        
        rect = plt.Rectangle((0.05, 0.025), width=0.9, height=y_pos - 0.025, zorder=4,
                             fill=True, facecolor='none', edgecolor='black', alpha=1.0,
                             lw=0.8)    
        
        ax.add_patch(rect)
        
        if gust:
            if not seasonal:
                speed_type = 'Wind Gust km/h (%)'
                fontsize=16
                pos = (-5.1, 1.07)
            else:
                speed_type = 'Wind Speed km/h'
                fontsize=6
                pos = (-2.1, 1.03)
        else:
            speed_type = 'Wind Speed km/h  (%)'
            pos = (-3.9, 1.07)
            fontsize=16
            
        prop = font_manager.FontProperties(fname=self.fname, size=fontsize)
            
        ax.text(*pos, 
                speed_type,
                ha='left', 
                fontproperties=prop)

        ax.axis('off')
        
        return ax
    
    def wind_rose(self, rosedata, speeds=None, gust=False):
        """
        """
        print('Creating Windrose Diagrams.')
        bar_dir, bar_width = self.wind_rose_plot_dirs()
        
        fig, ax = plt.subplots(figsize=(11, 9))
        
        fig.patch.set_alpha(0.0)
        
        ax.axis([-10, 10, -10, 10])
        
        ax.axis('off')
        
        ax_rose = fig.add_axes([0.05, 0.07, 0.65, 0.95], 
                               projection='polar', 
                               zorder=2)
    
        ax_rose.set_theta_direction('counterclockwise')
        ax_rose.set_theta_zero_location('N')
        ax_rose.set_rlabel_position(30)
        color_vals = []
    
        for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
            
            if n == 0:
                color_val = float((n) / (rosedata.shape[1] - 1))
                color_vals.append(color_val)
                
                ax_rose.bar(bar_dir, rosedata[c1].values, 
                            width=bar_width,
                            edgecolor='none',
                            color=self.cmap(color_val),
                            linewidth=0,)

            color_val = float((n + 1) / (rosedata.shape[1] - 1))
            color_vals.append(color_val)
            
            ax_rose.bar(bar_dir, rosedata[c2].values, 
                        width=bar_width, 
                        bottom=rosedata.cumsum(axis=1)[c1].values,
                        color=self.cmap(color_val),
                        edgecolor='none',
                        linewidth=0,
                        )
                
        cardinals = ['N', '', 'NW', '', 'W', '', 'SW', '', 
                     'S', '', 'SE', '', 'E', '', 'NE', '']
        percentages = ['{}%'.format(2 * x) for x in range(15)]
        
        prop = font_manager.FontProperties(fname=self.fname, size=22)
            
        ax_rose.set_rgrids([1, 2, 3, 4, 5, 6])
        ax_rose.tick_params(pad=8.0)
        ax_rose.set_xticks(np.linspace(0, 2 * np.pi, 17))
        ax_rose.set_xticklabels(cardinals,
                                fontproperties=prop,
                                zorder=7)
        
        prop = font_manager.FontProperties(fname=self.fname, size=20)
        
        ax_rose.set_yticks(np.arange(0, 100, 2))
        ax_rose.set_yticklabels(percentages, 
                                fontproperties=prop,
                                zorder=8)
            
        ax_rose.set_ylim(0, 1.0 * np.ceil(rosedata.sum(axis=1).max() / 1.0))
        
        ax_colorbar = fig.add_axes([0.77, 0.16, 0.05, 0.75], 
                                   zorder=1)
    
        ax_colorbar = self.color_bar(list(rosedata.columns),
                                     rosedata,
                                     ax_colorbar,
                                     gust,
                                     color_vals)
        
        prop = font_manager.FontProperties(fname=self.fname, size=24)
        
        if not gust:
            ax_colorbar.text(-6, -0.07, 
                             'Mean Wind Speed:', ha='left',
                             fontproperties=prop)
            ax_colorbar.text(3.7, -0.07, 
                             '{:.1F} km/h'.format(speeds[1]), ha='right',
                             fontproperties=prop)
        
            ax_colorbar.text(-6, -0.14, 
                             'Max Wind Speed:', ha='left',
                             fontproperties=prop)
            ax_colorbar.text(3.7, -0.14, 
                             '{:.1F} km/h'.format(speeds[0]), ha='right',
                             fontproperties=prop)
            
#        fig.savefig(out_file, dpi=240)
        
        return color_vals, rosedata.sum(axis=0).values
        
    def seasonal_windrose(self, summer, autumn, winter, spring, 
                          spd_bins, max_pc, color_vals, vals):
        """
        """
        print('Creating Seasonal Windrose Diagram.')
        cardinals = ['N', '', 'NW', '', 'W', '', 'SW', '', 
                     'S', '', 'SE', '', 'E', '', 'NE', '']
        bar_dir, bar_width = self.wind_rose_plot_dirs()
        rlabel_position = 30
    
    ###############################################################################
        # fig, main axis    
        fig, ax_main = plt.subplots(figsize=(30, 6))
        
        fig.patch.set_alpha(0.0)
        
        ax_main.axis([-10, 10, -10, 10])
        
        ax_main.axis('off')
        
        prop = font_manager.FontProperties(fname=self.fname, size=26)
            
        ax_main.text(-8.18, 9.0, 
                     'Summer', 
                     ha='center', 
                     fontproperties=prop)
        ax_main.text(-3.27, 9.0, 
                     'Autumn', 
                     ha='center', 
                     fontproperties=prop)
        ax_main.text(1.54, 9.0, 
                     'Winter', 
                     ha='center', 
                     fontproperties=prop)
        ax_main.text(6.2, 9.0, 
                     'Spring', 
                     ha='center', 
                     fontproperties=prop)
        
        prop = font_manager.FontProperties(fname=self.fname, size=22)
    
    ###############################################################################
        # summer axis
        ax_summer = fig.add_axes([-0.23, 0.06, 0.68, 0.8], 
                                 projection='polar', 
                                 zorder=2)
        
        ax_summer.set_theta_direction('counterclockwise')
        ax_summer.set_theta_zero_location('N')
        
        for n, (c1, c2) in enumerate(zip(summer.columns[:-1], summer.columns[1:])):
            if n == 0:
                color_val = float((n) / (summer.shape[1] - 1))
                
                ax_summer.bar(bar_dir, summer[c1].values, 
                              width=bar_width,
                              edgecolor='none',
                              color=self.cmap(color_val),
                              linewidth=0)
        
            color_val = float((n + 1) / (summer.shape[1] - 1))
        
            ax_summer.bar(bar_dir, summer[c2].values, 
                          width=bar_width, 
                          bottom=summer.cumsum(axis=1)[c1].values,
                          color=self.cmap(color_val),
                          edgecolor='none',
                          linewidth=0,
                          )

        ax_summer.tick_params(pad=8.0)
        ax_summer.set_xticks(np.linspace(0, 2 * np.pi, 17))
        ax_summer.set_xticklabels(cardinals,
                                  fontproperties=prop)
        
        ax_summer.set_rlabel_position(rlabel_position)
    
        ax_summer.set_ylim(0, max_pc)
        
        ytick_labels = ['{:.0F}'.format(x) + '%' for x in ax_summer.get_yticks()]
        
        ax_summer.set_yticklabels(ytick_labels, 
                                  fontproperties=prop)
        
    ###############################################################################
        # autumn axis    
        ax_autumn = fig.add_axes([0.02, 0.06, 0.65, 0.8], 
                                 projection='polar', 
                                 zorder=2)
        
        ax_autumn.set_theta_direction('counterclockwise')
        ax_autumn.set_theta_zero_location('N')
        
        for n, (c1, c2) in enumerate(zip(autumn.columns[:-1], autumn.columns[1:])):
            if n == 0:
                color_val = float((n) / (autumn.shape[1] - 1))
                
                ax_autumn.bar(bar_dir, autumn[c1].values, 
                              width=bar_width,
                              edgecolor='none',
                              color=self.cmap(color_val),
                              linewidth=0)
        
            color_val = float((n + 1) / (autumn.shape[1] - 1))
        
            ax_autumn.bar(bar_dir, autumn[c2].values, 
                          width=bar_width, 
                          bottom=autumn.cumsum(axis=1)[c1].values,
                          color=self.cmap(color_val),
                          edgecolor='none',
                          linewidth=0,
                          )
            
        ax_autumn.tick_params(pad=8.0)
        ax_autumn.set_xticks(np.linspace(0, 2 * np.pi, 17))
        ax_autumn.set_xticklabels(cardinals,
                                  fontproperties=prop)
                
        ax_autumn.set_rlabel_position(rlabel_position)
        
        ax_autumn.set_ylim(0, max_pc)
        
        ax_autumn.set_yticklabels(ytick_labels, 
                                  fontproperties=prop)
    
    ###############################################################################
        # winter axis    
        ax_winter = fig.add_axes([0.255, 0.06, 0.65, 0.8], 
                                 projection='polar', 
                                 zorder=2)
        
        ax_winter.set_theta_direction('counterclockwise')
        ax_winter.set_theta_zero_location('N')
        
        for n, (c1, c2) in enumerate(zip(winter.columns[:-1], winter.columns[1:])):
            if n == 0:
                color_val = float((n) / (winter.shape[1] - 1))
                
                ax_winter.bar(bar_dir, winter[c1].values, 
                              width=bar_width,
                              edgecolor='none',
                              color=self.cmap(color_val),
                              linewidth=0)
        
            color_val = float((n + 1) / (winter.shape[1] - 1))
        
            ax_winter.bar(bar_dir, winter[c2].values, 
                          width=bar_width, 
                          bottom=winter.cumsum(axis=1)[c1].values,
                          color=self.cmap(color_val),
                          edgecolor='none',
                          linewidth=0,
                          )
        
        ax_winter.tick_params(pad=8.0)
        ax_winter.set_xticks(np.linspace(0, 2 * np.pi, 17))
        ax_winter.set_xticklabels(cardinals,
                                  fontproperties=prop)
        
        ax_winter.set_rlabel_position(rlabel_position)
        
        ax_winter.set_ylim(0, max_pc)
            
        ax_winter.set_yticklabels(ytick_labels, 
                                  fontproperties=prop)
    
    ###############################################################################
        # spring axis    
        ax_spring = fig.add_axes([0.485, 0.06, 0.65, 0.8], 
                                 projection='polar', 
                                 zorder=2)
        
        ax_spring.set_theta_direction('counterclockwise')
        ax_spring.set_theta_zero_location('N')
        
        for n, (c1, c2) in enumerate(zip(spring.columns[:-1], spring.columns[1:])):
            if n == 0:
                color_val = float((n) / (spring.shape[1] - 1))
                
                ax_spring.bar(bar_dir, spring[c1].values, 
                              width=bar_width,
                              edgecolor='none',
                              color=self.cmap(color_val),
                              linewidth=0)
        
            color_val = float((n + 1) / (spring.shape[1] - 1))
        
            ax_spring.bar(bar_dir, spring[c2].values, 
                          width=bar_width, 
                          bottom=spring.cumsum(axis=1)[c1].values,
                          color=self.cmap(color_val),
                          edgecolor='none',
                          linewidth=0,
                          )
        
        ax_spring.tick_params(pad=8.0)
        ax_spring.set_xticks(np.linspace(0, 2 * np.pi, 17))
        ax_spring.set_xticklabels(cardinals,
                                  fontproperties=prop)
        
        ax_spring.set_rlabel_position(rlabel_position)
        
        ax_spring.set_ylim(0, max_pc)
        
        ax_spring.set_yticklabels(ytick_labels, 
                                  fontproperties=prop)
    
    ###############################################################################
        # colorbar axis
        prop = font_manager.FontProperties(fname=self.fname, size=18)
        
        ax_colorbar = fig.add_axes([0.92, 0.02, 0.02, 0.9],
                                   zorder=1)
        y_pos = 0.025

        labs = summer.columns.values

        for y_height, color_val, lab in zip(vals / 105, color_vals, labs):
            rect = plt.Rectangle((0.05, y_pos), width=0.9, height=y_height, zorder=2,
                                 fill=True, facecolor=self.cmap(color_val), edgecolor='darkgrey',
                                lw=0.01)
    
            if ('>' not in lab):
                corr = 0.0
                
                if (lab == 'calm') and (y_height * 105 < 2.0):
                    corr = 0.03
                    
                if (lab != 'calm') and (y_height * 105 < 2.0) and (color_val != 1.0):
                    corr = 0.01
                    
                ax_colorbar.text(1.05, 
                        (y_pos + 0.5 * y_height) - corr, 
                        lab.title() + ' ({:.1F}%)'.format(y_height * 105), 
                        va='center', 
                        fontproperties=prop)
                
            y_pos += y_height
            
            ax_colorbar.add_patch(rect)
            
        rect = plt.Rectangle((0.05, 0.025), width=0.9, height=y_pos - 0.025, zorder=3,
                             fill=True, facecolor='none', edgecolor='darkgrey', alpha=1.0,
                             lw=2.5)    
        
        ax_colorbar.add_patch(rect)
        
        rect = plt.Rectangle((0.05, 0.025), width=0.9, height=y_pos - 0.025, zorder=4,
                             fill=True, facecolor='none', edgecolor='black', alpha=1.0,
                             lw=1.5)    
        
        ax_colorbar.add_patch(rect)
        
        speed_type = 'Wind Speed km/h  (%)'
            
        ax_colorbar.text(-1.5, 1.03, 
                         speed_type,
                         ha='left', 
                         fontproperties=prop)

        ax_colorbar.axis('off')
        
        plt.tight_layout(pad=0.4)
        
#        fig.savefig(out_file, dpi=240)
                
    def color_bar(self, spd_labels, rosedata, ax, gust, color_vals):
        """
        """
        y_pos = 0.025
        
        labs = rosedata.columns.values
        vals = rosedata.sum(axis=0).values       
        
        prop = font_manager.FontProperties(fname=self.fname, size=18)

        for y_height, color_val, lab in zip(vals / 105, color_vals, labs):
            rect = plt.Rectangle((0.05, y_pos), width=0.9, height=y_height, zorder=2,
                                 fill=True, facecolor=self.cmap(color_val), edgecolor='darkgrey',
                                lw=0.01)
    
            if ('>' not in lab):
                corr = 0.0
                
                if (lab == 'calm') and (y_height * 105 < 2.0):
                    corr = 0.03
                    
                if (lab != 'calm') and (y_height * 105 < 2.0) and (color_val != 1.0):
                    corr = 0.01
                    
                ax.text(1.05, 
                        (y_pos + 0.5 * y_height) - corr, 
                        lab.title() + ' ({:.1F}%)'.format(y_height * 105), 
                        va='center', 
                        fontproperties=prop,)
                
            y_pos += y_height
            
            ax.add_patch(rect)
            
        rect = plt.Rectangle((0.05, 0.025), width=0.9, height=y_pos - 0.025, zorder=3,
                             fill=True, facecolor='none', edgecolor='darkgrey', alpha=1.0,
                             lw=2.5)    
        
        ax.add_patch(rect)
        
        rect = plt.Rectangle((0.05, 0.025), width=0.9, height=y_pos - 0.025, zorder=4,
                             fill=True, facecolor='none', edgecolor='black', alpha=1.0,
                             lw=1.5)    
        
        ax.add_patch(rect)
        
        if gust:
            speed_type = 'Wind Gust km/h (%)'
        else:
            speed_type = 'Wind Speed km/h  (%)'
            
        ax.text(-1.5, 1.03, 
                speed_type,
                ha='left', 
                fontproperties=prop)

        ax.axis('off')
        
        return ax
        
    def wind_rose_plot_dirs(self):
        """
        """    
        bar_dir = np.linspace(0, 2*np.pi, 17)[:-1]
        bar_width = ((2 * np.pi) / 16) - 0.01
        
        return bar_dir, bar_width

    def speed_labels(self, bins): 
        """
        """
        labels = []
        
        for left, right in zip(bins[:-1], bins[1:]):
            if left == bins[0]:
                labels.append('calm'.format(right))
            elif np.isinf(right):
                labels.append('>{:.0F}'.format(left))
            else:
                labels.append('{:.0F} - {:.0F}'.format(left, right))
    
        return list(labels) 
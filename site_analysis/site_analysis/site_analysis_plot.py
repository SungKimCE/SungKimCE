#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:28:44 2019

@author: shane
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import os
import logging

# custom SITE ANALYSIS libs
from site_analysis import windrose
from site_analysis import temp_rain
from site_analysis import humidity
from site_analysis import psychro
from site_analysis import clouds
from site_analysis import solar

helv_font = 'add_files/fonts/HelveticaNeueLTStd-Th.ttf'

class SiteAnalysisPlot:
    
    def __init__(self, lat, lng, stns_db, cmap, bom_dir, out_dir, file_name, address):
        """
        """
        print('#' * 60)
        print('Creating Site Analysis Plot.')
        logging.info('Begining Site Analysis Plot.')
        print('#' * 60)
        self.bom_dir = bom_dir
        self.wind_dir = os.path.join(self.bom_dir, 
                                     'wind_data_separated_into_station_number')
        
        self.prop = font_manager.FontProperties(fname=helv_font)
        self.fname = helv_font
        self.cmap = plt.get_cmap(cmap)
        self.axis_on_off = 'off'
                
        logging.info('Getting Station information.')
        self.stns = stns_db.stn_number.values[:3]
        self.weights = stns_db.weight.values[:3]
        self.elev = self.inverse_weight_diff(stns_db.elev.values[:3], 
                                             self.weights)
        
        self.location_fontsize = 10
        self.default_fontsize = 12
        self.heading_fontsize = 18
        
        # this part makes the plot
        logging.info('Creating Figure.')
        fig = plt.figure(figsize=(17, 11))
        
        self.main_axis(fig)
        self.title_axis(fig, lat, lng, address)
        self.wind_axes(fig)
        self.temp_axis(fig)        
        self.solar_axis(fig, lat, lng)
        self.psycho_axis(fig)
        
        print('#' * 60)
        print('Saving Site Analysis.')
        logging.info('Saving Site Analysis Figure.')
        print('#' * 60)   
    
#        fig.savefig(os.path.join(out_dir, file_name),
#                    dpi=240)
        print(out_dir)
        fig.savefig(os.path.join(out_dir, file_name.replace('.png', '.pdf')),
                    )

    def main_axis(self, fig):
        """
        """
        print('Creating Main axis.')
        logging.info('Creating the Main axis of the figure.')
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=10)
        ax.axis([0, 1, 0, 1])
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=self.heading_fontsize)
        
        # left lines around windrose title
        ax.axhline(0.92, 0.02, 0.15, zorder=10, color='darkgrey')
        ax.axhline(0.92, 0.35, 0.48, zorder=10, color='darkgrey')
        ax.text(0.25, 0.92, 'WINDROSE DIAGRAMS', 
                va='center', ha='center', 
                fontproperties=prop)
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=12)
        
        ax.text(0.25, 0.605, 'SEASONAL WIND SPEED', 
                va='center', ha='center', 
                fontproperties=prop)
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=self.heading_fontsize)
        
        # lines around Temp and rainfall title
        ax.axhline(0.92, 0.52, 0.63, zorder=10, color='darkgrey')
        ax.axhline(0.92, 0.88, 0.98, zorder=10, color='darkgrey')
        ax.text(0.75, 0.92, 'TEMPERATURE & RAINFALL', 
                ha='center', va='center',
                fontproperties=prop)
        
        # lines around solar title
        ax.axhline(0.4, 0.02, 0.15, zorder=10, color='darkgrey')
        ax.axhline(0.4, 0.35, 0.48, zorder=10, color='darkgrey')
        ax.text(0.25, 0.4, 'SOLAR DIAGRAMS', 
                ha='center', va='center',
                fontproperties=prop)
        
        # lines around psycometric title
        ax.axhline(0.4, 0.52, 0.63, zorder=10, color='darkgrey')
        ax.axhline(0.4, 0.88, 0.98, zorder=10, color='darkgrey')
        ax.text(0.75, 0.4, 'PSYCHROMETRIC DIAGRAM', 
                ha='center', va='center',
                fontproperties=prop)
        
        # vertical line dividing the page in half
        ax.axvline(0.5, 0.02, 0.9, zorder=10, color='darkgrey')
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=16)
                
        ax.text(0.15, 0.35, 'SOLAR IRRADIANCE', 
                ha='center', va='center',
                fontproperties=prop)
        
        ax.text(0.40, 0.35, 'SOLAR PATH', 
                ha='center', va='center',
                fontproperties=prop)
        
        ax.axis(self.axis_on_off)
        
        return ax
    
    def title_axis(self, fig, lat, lng, address):
        """
        """
        print('Creating Title axis.')
        logging.info('Creating the Title section axis.')
        ax_title = fig.add_axes([0.0, 0.90, 1.0, 0.1])
        ax_title.set_alpha(0.0)
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=28)
        
        import datetime
        string = address
        #string += ', Generated: {}'.format(datetime.datetime.now().strftime("%d/%m/%y"))
    
        ax_title.text(0.02, 0.6, 'SITE ANALYSIS ', zorder=5, 
                      fontproperties=prop)
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=16)
        
        ax_title.text(0.5, 0.6, string, zorder=5, ha='center',
                      fontproperties=prop)
        
     
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=self.location_fontsize)
        
        ax_title.text(0.02, 0.4, '\xa9' + 'URBANFINITY', 
                      zorder=5, 
                      fontproperties=prop)
        
        ax_title.text(0.98, 0.8, 'LOCATION INFORMATION', ha='right', 
                      zorder=5, fontproperties=prop)
        
        ax_title.text(0.98, 0.6, 'LAT: {:.4F} deg, LNG: {:.4F} deg'.format(lat, lng), 
                      ha='right', zorder=5, 
                      fontproperties=prop)
        
        ax_title.text(0.98, 0.4, 'BOM STATION NUMBERS: {}, {} and {}'.format(*self.stns),
                      ha='right', zorder=5, 
                      fontproperties=prop)
        
        ax_title.axis(self.axis_on_off)
        
        return ax_title
    
    def wind_axes(self, fig):
        """
        """
        print('Creating Windrose axes.')
        logging.info('Creating Windroses.')
        # gets the wind data and bins it combining it using the weighted inverse
        # of the distance method
        wind_data = windrose.WindData(self.stns, 
                                      self.wind_dir,
                                      self.weights,
                                      num_speed_bins=8)
        
        ###########################################################################
        # wind speed/gust windroses
        ax_wind_ul = fig.add_axes([-0.025, 0.64, 0.25, 0.25], projection='polar')
        ax_wind_ur = fig.add_axes([0.22, 0.64, 0.25, 0.25], projection='polar')
            
        ax_ul_color_bar = fig.add_axes([0.194, 0.65, 0.01, 0.22], zorder=6)
        ax_ur_color_bar = fig.add_axes([0.44, 0.65, 0.01, 0.22], zorder=6)
    
        # plots the windroses of the interpolated wind data.
        logging.info('Making Speed/Gust Windroses.')
        wind = windrose.WindRosePlot(wind_data,
                                     self.cmap,
                                     self.fname
                                     )
        
        ax_wind_ul = wind.wind_rose_ax(wind_data.wind_rose_data, 
                                       ax_wind_ul, 
                                       ax_ul_color_bar)
           
        ax_wind_ur = wind.wind_rose_ax(wind_data.wind_gust_data, 
                                       ax_wind_ur, 
                                       ax_ur_color_bar, 
                                       gust=True)
        
        ###########################################################################
        # seasonal windroses    
        ax_seasonal_color_bar = fig.add_axes([0.46, 0.42, 0.01, 0.18], zorder=6)
        
        seasonal_y_pos = 0.425
        seasonal__size = 0.14
        
        ###########################################################################
        # summer 
        logging.info('Making Summer Windrose.')
        ax_wind_summer = fig.add_axes([-0.01, seasonal_y_pos, 
                                       seasonal__size, seasonal__size], projection='polar')
        ax_wind_summer = wind.wind_rose_ax(wind_data.Summer, 
                                           ax_wind_summer, 
                                           ax_seasonal_color_bar, 
                                           seasonal_max_pc=wind_data.max_seasonal_pc,
                                           gust=True, 
                                           seasonal=True)
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=self.default_fontsize)
        
        ax_wind_summer[0].text(0, wind_data.max_seasonal_pc * 1.3, 
                      'Summer', ha='center', 
                      fontproperties=prop)
    
        ###########################################################################
        # Autumn 
        logging.info('Making Autumn Windrose.')
        ax_wind_autumn = fig.add_axes([0.105, seasonal_y_pos, 
                                       seasonal__size, seasonal__size], projection='polar')
        ax_wind_autumn = wind.wind_rose_ax(wind_data.Autumn, 
                                           ax_wind_autumn, 
                                           ax_color=None, 
                                           seasonal_max_pc=wind_data.max_seasonal_pc)
        
        ax_wind_autumn[0].text(0, wind_data.max_seasonal_pc * 1.3, 
                      'Autumn', ha='center', 
                      fontproperties=prop)
    
        ###########################################################################
        # spring     
        logging.info('Making Spring Windrose.')
        ax_wind_spring = fig.add_axes([0.22, seasonal_y_pos, 
                                       seasonal__size, seasonal__size], projection='polar')
        ax_wind_spring = wind.wind_rose_ax(wind_data.Spring, 
                                           ax_wind_spring, 
                                           ax_color=None, 
                                           seasonal_max_pc=wind_data.max_seasonal_pc)
        
        ax_wind_spring[0].text(0, wind_data.max_seasonal_pc * 1.3, 
                      'Spring', ha='center', 
                      fontproperties=prop)   
    
        ###########################################################################
        # winter    
        logging.info('Making Winter Windrose.')
        ax_wind_winter = fig.add_axes([0.335, seasonal_y_pos, 
                                       seasonal__size, seasonal__size], projection='polar')
        ax_wind_winter = wind.wind_rose_ax(wind_data.Winter, 
                                           ax_wind_winter, 
                                           ax_color=None, 
                                           seasonal_max_pc=wind_data.max_seasonal_pc)
        
        ax_wind_winter[0].text(0, wind_data.max_seasonal_pc * 1.3, 
                      'Winter', ha='center', 
                      fontproperties=prop)   

    def temp_axis(self, fig):
        """
        """
        print('Creating Temperature and Rainfall axes.')
        logging.info('Creating Temperature and Rainfall axes.')
        temp_rain_data = temp_rain.WeatherData(self.stns, 
                                               self.bom_dir, 
                                               self.weights)
        
        temp_rain_obj = temp_rain.Temp_Rainfall_Plot(temp_rain_data,
                                                     self.cmap,
                                                     self.fname)
    
        ax_temp = fig.add_axes([0.56, 0.76, 0.42, 0.13], zorder=6)
        ax_temp = temp_rain_obj.temp_plot_ax(ax_temp, 
                                             temp_rain_data.min_temps,
                                             temp_rain_data.min_temps_std,
                                             temp_rain_data.max_temps,
                                             temp_rain_data.max_temps_std
                                             )

        ax_rain = fig.add_axes([0.56, 0.58, 0.42, 0.13])
    
        ax_rain = temp_rain_obj.rain_plot_ax(ax_rain, temp_rain_data.rainfall)
        
        ###########################################################################
        # temp rain table
        
        temp_rain_obj.table_plot_ax(fig, 
                                    temp_rain_data.min_temps,
                                    temp_rain_data.min_temps_std,
                                    temp_rain_data.max_temps,
                                    temp_rain_data.max_temps_std,
                                    temp_rain_data.rainfall)
        
    def solar_axis(self, fig, lat, lng):
        """
        """ 
        print('Creating Solar Path and Solar Irradiance axes.')
        logging.info('Creating Solar Path and Solar Irradiance axes.')
        # gets the cloud data for each station and interpolates.
        cloud = clouds.CloudData(self.stns, 
                                 self.wind_dir, 
                                 self.weights)
        
        ax_solar_path = fig.add_axes([0.295, 0.02, 0.2, 0.3], zorder=2)        
        ax_solar_path.axis(self.axis_on_off)
        
        ax_solar_path.axis([-94, 94, -91, 91])
        
        solar.SolarPath(ax_solar_path, 
                        lat, 
                        lng, 
                        self.cmap,
                        self.prop)
        
        ax_solar_irr = fig.add_axes([0.04, 0.06, 0.22, 0.25], zorder=2)
    
        solar_irr = solar.SolarIrradiance(lat, 
                                          lng, 
                                          cloud, 
                                          self.cmap, 
                                          self.prop)
    
        solar_irr.plot_irr_ax(ax_solar_irr, solar_irr.irr, solar_irr.clouds)
    
    def psycho_axis(self, fig):
        """
        """
        print('Creating Psychrometric axis.')
        logging.info('Creating Psychrometric axis.')
        # gets humidity data and then interpolates based on the weights
        hum_data = humidity.HumidityData(self.stns, 
                                         self.wind_dir, 
                                         self.weights)
        
        ax_psychro = fig.add_axes([0.51, 0.05, 0.45, 0.33], zorder=3)
        
        # plots a psychrometric chart using the temperatures and humidities
        # from the hum_data object        
        psy = psychro.PsychroChart(ax_psychro,
                                   self.elev,
                                   hum_data.humidity.temp.values,
                                   hum_data.humidity.humidity.values,
                                   hum_data.humidity.weight.values,
                                   self.cmap,
                                   self.fname)
        
                
        import matplotlib.pyplot as plt
        
        
        ax_pie = fig.add_axes([0.57, 0.21, 0.18, 0.18], zorder=4)
        # Pie chart
        labels = ['Active Heating',
                  'Passive Heating',
                  'Humidification', 
                  'Comfort',
                  'Ventilation', 
                  'Evaporative Cooling', 
                  'Conditioning']
        
        psy.zone_counts['labels'] = labels
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=8)
        
        psy.zone_counts.sort_values(by='values', ascending=False, inplace=True)

        y_pos = np.arange(psy.zone_counts.shape[0])
        
        ax_pie.barh(y_pos, psy.zone_counts['values'] * 100, align='center', color=psy.zone_counts.cols)
        ax_pie.set_yticks(y_pos)
        ax_pie.set_yticklabels(psy.zone_counts.labels, fontproperties=prop)
        ax_pie.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
        ax_pie.invert_yaxis()  # labels read top-to-bottom
        
        ax_pie.set_xticklabels(['','','','','','',])
        ax_pie.set_xlabel('')
        
        for i, v in enumerate(psy.zone_counts['values'] * 100):
            if v < 0.1:
                ax_pie.text(v + 3, 
                            i + .25, 
                            '< 0.1 %'.format(v),  
                            fontproperties=prop)
            else:
                ax_pie.text(v + 3, 
                            i + .25, 
                            '{:.1F} %'.format(v),  
                            fontproperties=prop)
        
        ax_pie.set_alpha(0.0)
        ax_pie.set_facecolor((1,1,1,0))
        
        ax_pie.spines['top'].set_visible(False)
        ax_pie.spines['right'].set_visible(False)
        ax_pie.spines['bottom'].set_visible(False)
        ax_pie.spines['left'].set_visible(False)
        

                
#        patches, texts, autotexts = ax_pie.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%',
#                   startangle=90, pctdistance=0.65, explode = explode, 
#                   shadow = False)#, textprops=prop)
#        
#        for text in texts:
#            text.set_fontproperties(prop)
#            
#        for atext in autotexts:
#            atext.set_fontproperties(prop)
#        # Equal aspect ratio ensures that pie is drawn as a circle
#        ax_pie.axis('equal')
#        plt.tight_layout()
#        plt.show()
    
    def inverse_weight_diff(self, x, weights):
        """
        """
        return np.sum(weights * x) / np.sum(weights)
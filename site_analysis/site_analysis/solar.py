#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:13:23 2019

@author: shane
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude_fast, get_azimuth_fast
import pytz
import timezonefinder
import datetime

col_red = np.array([164, 0, 40]) / 255              # 1.0
col_dk_oragne = np.array([226, 75, 56]) / 255       # 0.858
col_orange = np.array([251, 163, 97]) / 255         # 0.715
col_yellow = np.array([254, 232, 162]) / 255        # 0.572
col_yblue = np.array([233, 245, 232]) / 255         # 0.429
col_lt_blue = np.array([162, 210, 228]) / 255       # 0.286
col_blue = np.array([88, 142, 191]) / 255           # 0.143
col_dk_blue = np.array([47, 59, 147]) / 255         # 0.0

class SolarPath:
    
    def __init__(self, ax, lat, lng, cmap, prop):
        """
        """
        print('Creating Solar Path Diagram.')
        self.cmap = plt.get_cmap(cmap)
        self.prop = prop
        self.ax = ax
        self.year = datetime.datetime.now().year
        self.lat = lat
        self.lng = lng
        
        tf = timezonefinder.TimezoneFinder()
        self.time_zone = pytz.timezone(tf.certain_timezone_at(lng=lng, lat=lat))
        
        self.plot_solar_path()
        
    def plot_solar_path(self):
        """
        """
#        fig = plt.figure(figsize=(10,10))
        
#        self.ax = fig.add_axes([0.06, 0.05, 0.92, 0.9])
#        self.ax.axis('off')
#        self.ax.axis([-94, 94, -91, 91])
        
        radial_axis_pos = -45
        
        # plots the radial axes
        self.plot_radial_axis(radial_axis_pos)
        
        # cardinal axis spokes
        self.plot_angular_axis()
        
        # Solstices : (times, alts, azs, xs, ys)
        winter = self.plot_solstice(6, self.cmap(0.5))
        summer = self.plot_solstice(12, self.cmap(0.5))
  
        # fill between
        self.fill_between_solstices(winter[3],
                                    winter[4], 
                                    summer[3], 
                                    summer[4])
            
        # calculate and print number of hours in the solstice days
        self.number_of_hours_in_day(self.cmap(0.5), *winter[:-2])
        
        self.number_of_hours_in_day(self.cmap(0.5), *summer[:-2])
            
        # hour labels
        self.hour_labels()
            
        # labels for the paths of the months other than the solstice
        self.monthly_labels()
        
#        fig.savefig(self.file_name, dpi=400)

    def plot_radial_axis(self, radial_axis_pos):
        """
        """
        ###############################################################################
        # Radius circles
        for r in [0, 15, 30, 45, 60, 75, 90]: #[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            thetas = np.linspace(0, 2 * np.pi, 100)
            rads = np.ones(thetas.shape) * r
            
            rad, phi = self.polar_2_cartesian(rads, thetas)
            
            line_color = 'darkgrey'
            _zorder = 2
            _lw = 0.3
            
            if r == 90:
                line_color = 'black'
                _lw = 4.0
                _zorder = 5
            
            self.ax.plot(rad, phi, color=line_color, lw=_lw, zorder=_zorder)
            
            theta = np.array([radial_axis_pos])
            
            x, y = self.polar_2_cartesian(r, theta)
            
            self.ax.text(x, y - 1.5,
                         r'{}$^{{\circ}}$'.format(90 - r), 
                         color='black',
                         #ha='center',
                         va='bottom',
                         fontproperties=self.prop,
                         fontsize=12,
                         zorder=7)
            
        for r in range(200):
            thetas = np.linspace(0, 2*np.pi, 100)
            rads = np.ones(thetas.shape) * ((r * 0.25) + 91.25)
            
            rad, phi = self.polar_2_cartesian(rads, thetas)
            
            self.ax.plot(rad, phi, color='white', lw=5.5, zorder=5)
        
    def plot_angular_axis(self):
        """
        """
        for i in range(17):
            theta = np.array([0, 0 + (i * (360 / 16))])
            r = np.array([0, 90])
            
            x, y = self.polar_2_cartesian(r, theta * (np.pi / 180))
            
            self.ax.plot(x, y, color='darkgrey', lw=0.3, zorder=2)
            
        cardinals = ['E', '', 'SE', '', 'S', '', 'SW', '', 
                     'W', '', 'NW', '', 'N', '', 'NE', '', ]
        
        for i in range(16):
            theta = np.array([0 - (i * (360 / 16))])
            r = np.array([94.5])
            
            x, y = self.polar_2_cartesian(r, theta * (np.pi / 180))
            
            self.ax.text(x, y, 
                         cardinals[i], 
                         color='black', 
                         zorder=6, 
                         ha='center',
                         va='center',
                         fontproperties=self.prop,
                         fontsize=12)
     
    def plot_solstice(self, month, solstice_color):
        """
            azs = azimuths which range from 0 to 90 where zero is the edge of the circle
            and 90 the center, however this is reversed for the axis hence we restrict
            the azimuths to < 90
        """
        
        times = pd.date_range(start='{}/21/2019'.format(month), 
                              end='{}/22/2019'.format(month), 
                              freq='1T',
                              tz=self.time_zone)[:-1]  
         
        alts = 90 - get_altitude_fast(self.lat, 
                                      self.lng, 
                                      times.values)
        
        azs = (get_azimuth_fast(self.lat, 
                                self.lng, 
                                times.values) + 90) * np.pi / 180.
        
        # converts angle, radius to x and y coords
        xs, ys = self.polar_2_cartesian(alts,
                                   azs)
        
        #ax.plot(xs_winter, ys_winter, color='black', lw=1.5, zorder=2)
        self.ax.plot(xs, ys, color=solstice_color, lw=1.8, zorder=3)
        
        return times, alts, azs, xs, ys
    
    def fill_between_solstices(self, xs_winter, ys_winter, xs_summer,
                               ys_summer):
        """
            fill between summer and winter solstice by filling winter circle, 
            then white out summer circle, need to figure out first whether 
            there is overlap between the two circles.
        """
        if self.find_overlap_in_arrays(ys_winter, ys_summer):
            # fill the winter circle with yellow and the summer with white
            self.ax.fill(xs_winter, ys_winter, color=self.cmap(0.572), alpha=1.0)
            self.ax.fill(xs_summer, ys_summer, color='white', alpha=1.0)
            
        else:
            # fills the entire plot space with the fill color
            self.ax.fill([-191, -191, 191, 191],
                    [-191, 191, -191, 191],
                    color=self.cmap(0.572))
            self.ax.fill([-91, 91, -91, 91],
                    [-91, -91, 91, 91],
                    color=self.cmap(0.572))
            
            # fill the circles with white so what is left over is the solar path.
            self.ax.fill(xs_winter, ys_winter, color='white', alpha=1.0)
            self.ax.fill(xs_summer, ys_summer, color='white', alpha=1.0)
        
    def number_of_hours_in_day(self, col, times, alts, azs):
        """
            number of hours in the day/ dusk and dawn of the solstices
        """
        xs, ys = self.polar_2_cartesian(alts[alts < 90],
                                        azs[alts < 90])
        
        hours = times[alts < 90][-1] - times[alts < 90][0]
        
        # if we want to put dawn and dusk times instead of total 
        # length of the solstice
        #dawn = times[alts[0] < 90][0]
        #dusk = times[alts[0] < 90][-1]
    
        self.ax.text(xs[0] - 0, ys[0] + 2.0,
                     '{:.0F} hr {:.0F} m'.format(*self.hours_minutes(hours)), 
                     color='black', 
                     fontproperties=self.prop, 
                     fontsize=10, ha='right', 
                     zorder=11)
        
    def hour_labels(self):
        """
            summer/winter hour labels
        """
        for month, col in zip([6, 12], ['black', 'black']):
            hours = pd.date_range(start='{}/21/2019'.format(month), 
                                  end='{}/22/2019'.format(month), 
                                  freq='3H',
                                  tz=self.time_zone)
        
            print(hours)
            labels = hours.strftime('%-I%p')
            print(labels)
         
            alts = 90 - get_altitude_fast(self.lat, self.lng, hours.values)
            
            azs = (get_azimuth_fast(self.lat, self.lng, hours.values) + 90) * np.pi / 180.
        
            print(alts, azs * 180 / np.pi)
            sel = (alts <= 88)
            
            xs, ys = self.polar_2_cartesian(alts[sel], azs[sel])
            
            self.ax.scatter(xs * -1, ys, marker='o', color=col, s=10, zorder=5)
            
            if month == 6:    
                ys += 8.0
            else:
                ys -= 9.0
            
            for i in range(labels[sel].shape[0]):
                print(xs[i] * -1, ys[i], labels[sel][i])
                self.ax.text(xs[i] * -1, ys[i], 
                             labels[sel][i], 
                             color=col, 
                             ha='center',
                             va='center',
                             fontproperties=self.prop,
                             fontsize=12,
                             zorder=6)
                
    def monthly_labels(self):
        """
            months between winter and summer solstice
            in order of months after solstices for plotting purposes
        """
        for month, lab in zip([1, 2, 3, 4, 5], 
                              ['Jan/Nov', 'Feb/Oct', 'Mar/Sep', 
                               'Apr/Aug', 'May/Jul']):
            hours = pd.date_range(start='{}/21/2019'.format(month), 
                                  end='{}/22/2019'.format(month), 
                                  freq='1T',
                                  tz=self.time_zone)
         
            alts = 90 - get_altitude_fast(self.lat, self.lng, hours.values)
            
            azs = (get_azimuth_fast(self.lat, self.lng, hours.values) + 90) * \
            np.pi / 180.
        
            sel = alts <= 88
            
            xs, ys = self.polar_2_cartesian(alts[sel], azs[sel])
            
            self.ax.plot(xs, ys, color=self.cmap(0.858), lw=0.5, zorder=3, ls='--')
        
            # get the angle of the line between 80 and 90 on the right hand side of the
            # plot for the angel of rotation of the text        
            sel = (alts > 70) & (alts < 80)
            
            xs, ys = self.polar_2_cartesian(alts[sel], azs[sel])
        
            ys = ys[xs > 0]
            xs = xs[xs > 0]
            
            grad = (ys[-15] - ys[0]) / (xs[-15] - xs[0])
            grad_inc = np.arctan(grad)
            
            sel = alts < 70
                
            # plot the text at the positions near the 70 deg line
            xs, ys = self.polar_2_cartesian(alts[sel], azs[sel])
            
            if len(xs) != 0:    
                self.ax.text(xs[-1], ys[-1], 
                             '{}'.format(lab), 
                             color='black',
                             zorder=6, 
                             fontproperties=self.prop,
                             fontsize=10,
                             rotation=360 + (grad_inc * 180. / np.pi) + 5,)
    
    def cartesian_2_polar(self, x, y):
        """
        """
        radius = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return radius, phi
    
    def polar_2_cartesian(self, radius, phi):
        """
        """
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        return x, y
    
    def find_nearest_but_not_exceeding(self, array, arr2, value):
        """
        """
        new_array = np.asarray(array)
        new_array = new_array[(new_array < value) & (arr2 < 2.0)]
        idx = (np.abs(new_array - value)).argmin()
        return np.where(array == new_array[idx])[0][0]
    
    def find_overlap_in_arrays(self, arr1, arr2):
        """
        """
        for i in arr1:
            if any(i < arr2):
                return True
    
        return False

    def hours_minutes(self, td):
        return td.seconds//3600, (td.seconds//60)%60

class SolarIrradiance:
    
    def __init__(self, lat, lng, cloud_data, cmap, prop):
        """
        """
        print('Creating Solar Irradiance Plot.')
        self.cmap = plt.get_cmap(cmap)
        self.prop = prop
        tf = timezonefinder.TimezoneFinder()
        self.time_zone = pytz.timezone(tf.certain_timezone_at(lng=lng, lat=lat))
        
        date_rng = pd.date_range(start='1/1/{}'.format(datetime.datetime.now().year), 
                                 end='1/1/{}'.format(datetime.datetime.now().year + 1), 
                                 freq='H',
                                 tz=self.time_zone)[:-1]

        alts = get_altitude_fast(lat, lng, date_rng.values)

        irradiances = np.array([get_radiation_direct(date, alt) 
        if alt > 0 else 0
        for date, alt in zip(date_rng, alts)])

        irr = pd.DataFrame({'Month': date_rng.month, 
                            'Day': date_rng.day, 
                            'irradiance': irradiances
                            })

        irr_by_day = irr.groupby(['Month', 'Day']).irradiance.sum() / 1000.0
        
        irr = pd.DataFrame({'Mean': irr_by_day.groupby('Month').mean(),
                            'Std': irr_by_day.groupby('Month').std()
                            })
    
        self.irr = irr
        
        self.clouds = cloud_data.clouds
        
#        self.plot_solar_irradiance(clouds, irr)
        
    def plot_irr_ax(self, ax, irr, clouds):
        """
        """
        ax.plot(irr.index,
                irr.Mean,
                color=self.cmap(0.75),
                alpha=1, 
                lw=1.2,
                label='Solar Irradiance')
        
        ax.plot(irr.index,
                irr.Mean + 2 * irr.Std,
                color=self.cmap(0.875),
                alpha=1,
                ls='--',
                lw=0.9)
        
        ax.plot(irr.index,
                irr.Mean - 2 * irr.Std,
                color=self.cmap(0.625),
                alpha=1.0,
                ls='--',
                lw=0.9)
        
        ax.fill_between(irr.index,
                        irr.Mean + 2 * irr.Std,
                        irr.Mean - 2 * irr.Std,
                        color=self.cmap(0.875),
                        alpha=0.6)
        clouds /= 8
        
        solar_irr_minus_cloud = (1 - clouds.Mean) * irr.Mean
        solar_irr_minus_cloud_std = np.sqrt(clouds.Std**2 + irr.Std**2)
        
        df = pd.DataFrame({'Mean': solar_irr_minus_cloud, 
                           'Std': solar_irr_minus_cloud_std
                           })
        
        ax.plot(df.index,
                df.Mean,
                color=self.cmap(0.25),
                alpha=1.0,
                lw=1.2,
                label='Corrected Solar Irradiance')
        
        ax.plot(df.index,
                df.Mean + 2 * df.Std,
                color=self.cmap(0.375),
                alpha=1.0,
                ls='--',
                lw=0.9)
        
        ax.plot(df.index,
                df.Mean - 2 * df.Std,
                color=self.cmap(0.125),
                alpha=1.0,
                ls='--',
                lw=0.9)
        
        ax.fill_between(df.index,
                        df.Mean + 2 * df.Std,
                        df.Mean - 2 * df.Std,
                        color=self.cmap(0.25),
                        alpha=0.6)
        
        ax.legend(loc='best', prop=self.prop, fontsize=10)
        
        ax.set_xlabel('Month', fontproperties=self.prop, fontsize=12)
        ax.set_xticks(np.arange(1, 13, 1))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
        fontproperties=self.prop, fontsize=12)
        
        ax.set_ylabel(r'Solar Irradiance kW hr per sqr m per day', 
                      fontproperties=self.prop, fontsize=12)
        ax.set_yticks(np.arange(0, 30, 2))
        ax.set_yticklabels(np.arange(0, 30, 2), fontproperties=self.prop, fontsize=12)
        ax.set_ylim(0, 14.0)
                
        ax.grid(zorder=1)
        ax.tick_params(pad=1.0, grid_lw=0.2)
        
        return ax

    def plot_solar_irradiance(self, clouds, irr):
        """
        """
        fig, ax = plt.subplots(figsize=(11, 6))
        
        fig.set_alpha(0.0)
        fig.patch.set_alpha(0.0)
        ax.set_alpha(0.0)
           
        ax.plot(irr.index,
                irr.Mean,
                color=self.cmap(0.75),
                alpha=1, 
                lw=1.2,
                label='Solar Irradiance')
        
        ax.plot(irr.index,
                irr.Mean + 2 * irr.Std,
                color=self.cmap(0.875),
                alpha=1,
                ls='--',
                lw=0.9)
        
        ax.plot(irr.index,
                irr.Mean - 2 * irr.Std,
                color=self.cmap(0.625),
                alpha=1.0,
                ls='--',
                lw=0.9)
        
        ax.fill_between(irr.index,
                        irr.Mean + 2 * irr.Std,
                        irr.Mean - 2 * irr.Std,
                        color=self.cmap(0.875),
                        alpha=0.6)

        clouds /= 8
        
        solar_irr_minus_cloud = (1 - clouds.Mean) * irr.Mean
        solar_irr_minus_cloud_std = np.sqrt(clouds.Std**2 + irr.Std**2)
        
        df = pd.DataFrame({'Mean': solar_irr_minus_cloud, 
                           'Std': solar_irr_minus_cloud_std
                           })
        
        ax.plot(df.index,
                df.Mean,
                color=self.cmap(0.25),
                alpha=1.0,
                lw=1.2,
                label='Corrected Solar Irradiance')
        
        ax.plot(df.index,
                df.Mean + 2 * df.Std,
                color=self.cmap(0.375),
                alpha=1.0,
                ls='--',
                lw=0.9)
        
        ax.plot(df.index,
                df.Mean - 2 * df.Std,
                color=self.cmap(0.125),
                alpha=1.0,
                ls='--',
                lw=0.9)
        
        ax.fill_between(df.index,
                        df.Mean + 2 * df.Std,
                        df.Mean - 2 * df.Std,
                        color=self.cmap(0.25),
                        alpha=0.6)
        
        ax.set_ylim(0, 14.0)
        
        ax.legend(loc='best', fontsize=14, prop=self.prop)
        
        ax.set_xlabel('Month', fontproperties=self.prop, fontsize=14)
        ax.set_xticks(np.arange(1, 13, 1))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontproperties=self.prop, fontsize=14)
        
        ax.set_ylabel(r'Solar Irradiance kW hr per sqr m per day', fontproperties=self.prop, fontsize=14)
        
        ax.grid(zorder=1)
        
        fig.savefig(self.file_name, dpi=300)
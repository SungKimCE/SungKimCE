#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:02:52 2019

@author: shane
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

from psychrochart.chart import PsychroChart as ps

col_red = np.array([164, 0, 40, 255]) / 255
col_dk_oragne = np.array([226, 75, 56]) / 255
col_orange = np.array([251, 163, 97, 75]) / 255
col_yellow = np.array([254, 232, 162]) / 255
col_yblue = np.array([233, 245, 232]) / 255
col_lt_blue = np.array([162, 210, 228]) / 255
col_blue = np.array([88, 142, 191, 75]) / 255
col_dk_blue = np.array([47, 59, 147, 255]) / 255

class PsychroChart:
    
    def __init__(self, ax_psychro, elevation, temps, humidities, 
                 weights, cmap, font_name):
        """
        """
        print('Creating Psychrochart.')
        
        self.cmap = plt.get_cmap(cmap)
        self.fname = font_name
        self.elev = elevation
        
        custom_style = self.chart_style()
                
        base = 2.5
        
        temp_range = [base * np.floor(temps.min() / base),
                      base * np.ceil(temps.max() / base)]
        
        custom_style['limits']['altitude_m'] = elevation
        custom_style['limits']['range_temp_c'] = temp_range
            
        # create the chart with the custom style
        self.chart = ps(custom_style)
        
        zones_conf = self.chart_zones(temp_range[0])
            
        # zones must be appended before plotting
        self.chart.append_zones(zones_conf)     
         
        self.chart.plot(ax=ax_psychro)   
            
#        const_hum = (self.chart.axes.get_lines()[-1].get_xdata(), 
#                     self.chart.axes.get_lines()[-1].get_ydata())
        
        #self.fix_chart_axes(*const_hum)
        
        #self.vertical_comfort_temps(12, [22, 27], 32)

        # points are plotted after the chart itself
        points = self.make_points_dict(temps,
                                       humidities,
                                       weights)
                
        self.point_dict = self.chart.plot_points_dbt_rh(points)
        
                
        hum_ratios = self.get_hum_ratio(temps)
        
        self.count_zones(temps, hum_ratios, humidities)
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=14)
        
        self.chart.axes.xaxis.set_label_text(r'Dry-Bulb Temperature [$^{\circ}$C]',
                                             fontproperties=prop)
        
        self.chart.axes.yaxis.set_label_text(r'Humidity Ratio', # $\left[ g_w / kg_{da} \right]$',
                                             fontproperties=prop)

        active_heat_x_pos = temp_range[0] + ((10 - temp_range[0]) / 2.0)
        
        if active_heat_x_pos  < 10:
            self.chart.axes.text(active_heat_x_pos, 2, 'Active Heating', 
                                 color='black', 
                                 ha='center',
                                 fontproperties=prop)
        
        self.chart.axes.text(15, 6, 'Passive Heating', 
                             color='black', 
                             ha='center',
                             fontproperties=prop)
        
        self.chart.axes.text(17, 0.5, 'Humidification', 
                             color='black', 
                             ha='center',
                             fontproperties=prop)
        
        self.chart.axes.text(23, 9, 'Comfort', 
                             color='black', 
                             ha='center',
                             fontproperties=prop)
        
        self.chart.axes.text(27, 18, 'Ventilation', 
                             color='black', 
                             ha='center',
                             fontproperties=prop)
        
        self.chart.axes.text(34, 2, 'Evaporative Cooling', 
                             color='black', 
                             ha='center',
                             fontproperties=prop)
        
        self.chart.axes.text(37, 24, 'Conditioning', 
                             color='black', 
                             ha='center',
                             fontproperties=prop)
        
        self.get_hum_ratios(np.arange(20, 27, 0.5), 0.2)

    def get_hum_ratios(self, dbts, hum):
        """
        """
        from psychrochart import equations as eq
        
        press = eq.pressure_by_altitude(self.elev)
        
        wets = [eq.wet_bulb_temperature(dry_temp_c=dbt, relative_humid=hum, p_atm_kpa=press) for dbt in dbts]
        
        hum_rats = [(eq.humidity_ratio_from_temps(dry_bulb_temp_c=dbt, wet_bulb_temp_c=wbt, p_atm_kpa=press) * 1000) for dbt, wbt in zip(dbts, wets)]
        
        return hum_rats
        
    def chart_style(self):
        """
        """
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=12)
                
        return {
                "figure": {"figsize": [15, 8],
                           "base_fontsize": 12,
                           "title": "",
                           "x_label": r'Dry-Bulb Temperature [$^{\circ}C$]',
                           "y_label": r'Humidity Ratio $\left[ g_w / kg_{da} \right]$',
                           "partial_axis": True,
                           "fontproperties": prop,
                           "alpha": 0.0
                },
                "limits": {"range_temp_c": [-10, 50],
                           "range_humidity_g_kg": [0, 30],
                           "altitude_m": 40,
                           "step_temp": 0.1
                },
                "saturation": {"color": 'black',            # 100% RH
                               "linewidth": 1.2,
                               "zorder": 4},
                "constant_rh": {"color": [0.0, 0.0, 0.0, .4],
                                "linewidth": 1.0,           # RH lines
                                "linestyle": (2, (10, 5))
                                },
                "constant_dry_temp": {"color": [0.66, 0.66, 0.66, .7],
                                      "linewidth": 1.0,
                                      "linestyle": ":"},
                "constant_humidity": {"color": [0.66, 0.66, 0.66, .7],
                                      "linewidth": 1.0,
                                      "linestyle": ":"},
                "chart_params": {"with_constant_rh": True,
                                 "constant_rh_curves": [0, 20, 40, 60, 80, 100],
                                 "constant_rh_labels": [0, 20, 40, 60, 80, 100],
                                 "constant_rh_labels_loc": 0.73,
                                 "with_constant_v": False,
                                 "with_constant_h": False,
                                 "with_zones": False,
                                 "with_constant_wet_temp": False,
                                 "constant_wet_temp_label": 'Constant wet bulb temperature',
                                 "constant_wet_temp_step": 1,
                                 "range_wet_temp": [-10, 45],
                                 "constant_wet_temp_labels": [0, 10, 20, 30],
                                 "constant_wet_temp_labels_loc": 0.1,
                                 "with_constant_dry_temp": True,
                                 "constant_temp_label": 'Dry bulb temperature',
                                 "constant_temp_step": 2.5,
                                 "constant_temp_label_step": 5,
                                 "constant_temp_label_include_limits": True,
                                 "constant_temp_color": [0.66, 0.66, 0.66, .7],
                                 "constant_humid_label": 'Absolute humidity',
                                 "constant_humid_step": 2.0,
                                 "constant_humid_label_step": 5,
                                 "constant_humid_label_include_limits": True,
                }
            }
                
    def chart_zones(self, min_temp):
        """
        """
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=8)
        
        alpha_val = 0.27
        
        col_00 = self.cmap(0.0)
        col_00_al = list(col_00)
        col_00_al[-1] = alpha_val
        
        col_15 = self.cmap(0.15)
        col_15_al = list(col_15)
        col_15_al[-1] = alpha_val
        
        col_30 = self.cmap(0.30)
        col_30_al = list(col_30)
        col_30_al[-1] = alpha_val
        
        col_45 = self.cmap(0.45)
        col_45_al = list(col_45)
        col_45_al[-1] = alpha_val
        
        col_60 = self.cmap(0.60)
        col_60_al = list(col_60)
        col_60_al[-1] = alpha_val
        
        col_75 = self.cmap(0.75)
        col_75_al = list(col_75)
        col_75_al[-1] = alpha_val
        
        col_90 = self.cmap(0.90)
        col_90_al = list(col_90)
        col_90_al[-1] = alpha_val
        
        back_color = (0,0,0)
        
        return {"zones":[{"zone_type": "dbt-rh",
                         "style": {"edgecolor": back_color,
                                   "facecolor": 'none',
                                   "linewidth": 2.0,
                                   "linestyle": "-",
                                   "zorder": 5},
                         "points_x": [min_temp, 10],
                         "points_y": [0, 100],
                         "label_loc": 0.1,
#                         "label": "Active Heating"
                         },
                        {"zone_type": "dbt-rh",
                         "style": {"edgecolor": col_00,
                                   "facecolor": col_00_al,
                                   "linewidth": 1.5,
                                   "linestyle": "--",
                                   "zorder": 6},
                         "points_x": [min_temp, 10],
                         "points_y": [0, 100],
                         "label_loc": 0.1,
#                         "label": "Active Heating"
                         },
                         {"zone_type": "dbt-rh",
                          "style": {"edgecolor": back_color,
                                    "facecolor": 'none',
                                    "linewidth": 2.0,
                                    "linestyle": "-",
                                    "zorder": 5},
                         "points_x": [10, 20],
                         "points_y": [20, 100],
#                         "label": "Passive Heating",
                         "label_loc": 0.1,
                         },
                        {"zone_type": "dbt-rh",
                          "style": {"edgecolor": col_15,
                                    "facecolor": col_15_al,
                                    "linewidth": 1.5,
                                    "linestyle": "--",
                                    "zorder": 6},
                         "points_x": [10, 20],
                         "points_y": [20, 100],
#                         "label": "Passive Heating",
                         "label_loc": 0.1,
                         },
                    {"zone_type": "xy-points",
                          "style": {"edgecolor": back_color,
                                    "facecolor": 'none',
                                    "linewidth": 2.0,
                                    "linestyle": "-",
                                    "zorder": 5},
                         "points_x": [10.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
                                      13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
                                      16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5,
                                      20.0, 26.66, 10.0],
                         "points_y": [0.0] + self.get_hum_ratios(np.arange(10.0, 20.5, 0.5), 0.2) + [0.0, 0.0],
#                         "label": "Humidification",
                         "label_loc": -1.3,
                    },
                    {"zone_type": "xy-points",
                          "style": {"edgecolor": col_30,
                                    "facecolor": col_30_al,
                                    "linewidth": 1.5,
                                    "linestyle": "--",
                                    "zorder": 6},
                         "points_x": [10.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
                                      13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
                                      16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5,
                                      20.0, 26.66, 10.0],
                         "points_y": [0.0] + self.get_hum_ratios(np.arange(10.0, 20.5, 0.5), 0.2) + [0.0, 0.0],
#                         "label": "Humidification",
                         "label_loc": -1.3,
                    },
                         {"zone_type": "xy-points",
                          "style": {"edgecolor": back_color,
                                    "facecolor": 'none',
                                    "linewidth": 2.0,
                                    "linestyle": "-",
                                    "zorder": 5},
                         "points_x": [20.0] + list(np.arange(20.0, 24.32, 0.5)) + [26.66] + list(np.arange(26.66, 20.0, -0.5)) + [20.0],
                         "points_y": self.get_hum_ratios([20.0], 0.2) + self.get_hum_ratios(np.arange(20.0, 24.32, 0.5), 0.8) + [11.0] + self.get_hum_ratios(np.arange(26.66, 20.0, -0.5), 0.2) + self.get_hum_ratios([20.0], 0.2),
#                         "label": "Comfort",
                         "label_loc": 0.1,
                         "label_properties": {"fontproperties": prop}
                         },
                        {"zone_type": "xy-points",
                          "style": {"edgecolor": col_45,
                                    "facecolor": col_45_al,
                                    "linewidth": 1.5,
                                    "linestyle": "--",
                                    "zorder": 15},
                         "points_x": [20.0] + list(np.arange(20.0, 24.32, 0.5)) + [26.66] + list(np.arange(26.66, 20.0, -0.5)) + [20.0],
                         "points_y": self.get_hum_ratios([20.0], 0.2) + self.get_hum_ratios(np.arange(20.0, 24.32, 0.5), 0.8) + [11.0] + self.get_hum_ratios(np.arange(26.66, 20.0, -0.5), 0.2) + self.get_hum_ratios([20.0], 0.2),
#                         "label": "Comfort",
                         "label_loc": 0.1,
                         "label_properties": {"fontproperties": prop}
                         },
                    {"zone_type": "xy-points",
                          "style": {"edgecolor": back_color,
                                    "facecolor": 'none',
                                    "linewidth": 2.0,
                                    "linestyle": "-",
                                    "zorder": 5},
                         "points_x": [20.] + list(np.arange(20.0, 28.37, 0.5)) + [32.22, 32.22] + list(np.arange(32.22, 26.16, -0.5)),
                         "points_y": self.get_hum_ratios([20.0], 0.2) + self.get_hum_ratios(np.arange(20.0, 28.37, 0.5), 1.0) + [15.5, 6.24] + self.get_hum_ratios(np.arange(32.22, 26.16, -0.5), 0.2),
#                         "label": "Ventilation",
                         #"label_loc": 20.0,
                    },
                                
                    {"zone_type": "xy-points",
                          "style": {"edgecolor": col_75,
                                    "facecolor": col_75_al,
                                    "linewidth": 1.5,
                                    "linestyle": "--",
                                    "zorder": 6},
                         "points_x": [20.] + list(np.arange(20.0, 28.37, 0.5)) + [32.22, 32.22] + list(np.arange(32.22, 26.16, -0.5)),
                         "points_y": self.get_hum_ratios([20.0], 0.2) + self.get_hum_ratios(np.arange(20.0, 28.37, 0.5), 1.0) + [15.5, 6.24] + self.get_hum_ratios(np.arange(32.22, 26.16, -0.5), 0.2),
                         #"label": "Ventilation",
                         #"label_loc": 20.0,
                    },
                    {"zone_type": "xy-points",
                          "style": {"edgecolor": back_color,
                                    "facecolor": 'none',
                                    "linewidth": 2.0,
                                    "linestyle": "-",
                                    "zorder": 5},
                         "points_x": list(np.arange(20.0, 26.66, 0.5)) + [26.66, 26.66, 24.0, 38.33, 40.55, 40.55, 26.66, 20.0],
                         "points_y": self.get_hum_ratios(np.arange(20.0, 26.66, 0.5), 0.2) + self.get_hum_ratios([26.66], 0.2) + [11.0] + self.get_hum_ratios([24.0], 0.8) + [9.2, 6.1, 0.0, 0.0, 2.8],
#                         "label": "Evaporative Cooling",
                         "label_loc": 5.0,
                         "label_properties": {'zorder': 15},
                         "zorder": 15
                    },
                    {"zone_type": "xy-points",
                          "style": {"edgecolor": col_90,
                                    "facecolor": col_90_al,
                                    "linewidth": 1.5,
                                    "linestyle": "--",
                                    "zorder": 6},
                         "points_x": list(np.arange(20.0, 26.66, 0.5)) + [26.66, 26.66, 24.0, 38.33, 40.55, 40.55, 26.66, 20.0],
                         "points_y": self.get_hum_ratios(np.arange(20.0, 26.66, 0.5), 0.2) + self.get_hum_ratios([26.66], 0.2) + [11.0] + self.get_hum_ratios([24.0], 0.8) + [9.2, 6.1, 0.0, 0.0, 2.8],
#                         "label": "Evaporative Cooling",
                         "label_loc": -1.3,
                    },
#                        {"zone_type": "dbt-rh",
#                          "style": {"edgecolor": col_red,
#                                    "facecolor": col_orange,
#                                    "linewidth": 1.5,
#                                    "linestyle": "--",
#                                    "zorder": 6},
#                         "points_x": [20, 24.0],
#                         "points_y": [20, 100],
#                         "label": "Winter",
#                         "label_loc": 0.1,
#                         },
#                        {"zone_type": "dbt-rh",
#                         "style": {"edgecolor": [0.498, 0.624, 0.8],
#                                   "facecolor": [0.498, 0.624, 1.0, 0.2],
#                                   "linewidth": 1.5,
#                                   "linestyle": "--"},
#                         "points_x": [2, 20],
#                         "points_y": [0, 100],
#                         "label": "Winter"
#                         },
                        ]
                    }
                    
    def vertical_comfort_temps(self, t_min, t_opt, t_max):
        """
        """
        s_min = {"color": col_dk_blue, "lw": 2, "ls": ':', 'alpha': 0.6}
        #s_opt = {"color": col_yellow, "lw": 2, "ls": ':', 'alpha': 1.0}
        s_max = {"color": col_red, "lw": 2, "ls": ':', 'alpha': 0.6}
        
        l_min = r'Too Cold (${}^{{\circ}}C$)'.format(t_min)
        l_max = r'Too Hot (${}^{{\circ}}C$)'.format(t_max)
        
        prop = font_manager.FontProperties(fname=self.fname, 
                                           size=12)
        
        self.chart.plot_vertical_dry_bulb_temp_line(
                t_min, s_min, l_min, 
                ha='left', loc=0.05, 
                fontsize=12, alpha=0.4, fontproperties=prop)
        
#        for opt_t in t_opt:
#            self.chart.plot_vertical_dry_bulb_temp_line(opt_t, s_opt)
        
        self.chart.plot_vertical_dry_bulb_temp_line(
                t_max, s_max, l_max, 
                ha='left', loc=0.015, 
                fontsize=12, alpha=0.4, fontproperties=prop)

    def fix_chart_axes(self, temperature, humidity):
        """
        """
        # turn off the axis lines
        self.chart.axes.spines['left'].set_visible(False)
        self.chart.axes.spines['right'].set_visible(False)
        self.chart.axes.spines['bottom'].set_visible(False)
        self.chart.axes.spines['top'].set_visible(False)
        
        y_max = humidity[0] / self.chart.axes.get_ylim()[1]
        
        y_lim = self.chart.axes.get_ylim()[1]
        idx = self.find_nearest(humidity, y_lim)
        
        coord_1 = (temperature[idx - 1], humidity[idx - 1])
        coord_2 = (temperature[idx], humidity[idx])
        
        # these are the coords on the 100% rh line above and below the intersection
        # with the max y axis
        coords = [coord_1, coord_2]
        
        x_coords, y_coords = zip(*coords)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        
        x_min = (y_lim - c) / m
        
        neg = self.chart.axes.get_xlim()[0]
        
        if neg < 0:
            neg = abs(neg)
        else:
            neg = -1 * neg
        
        # left axis line
        self.chart.axes.axvline(x=self.chart.axes.get_xlim()[0] + 0.01,
                                ymin=0, ymax=y_max - 0.02 * y_max, color='black', lw=3.5)
        # right axis line
        self.chart.axes.axvline(x=self.chart.axes.get_xlim()[1], 
                                ymin=0, ymax=1.0, color='black', lw=3.5)
        # top axis line
        x_min = (x_min + neg) / (self.chart.axes.get_xlim()[1] - self.chart.axes.get_xlim()[0])
        
        self.chart.axes.axhline(y=y_lim, xmin=x_min, xmax=1.0, color='black', lw=3.5)
        # bottom axis line
        self.chart.axes.axhline(y=self.chart.axes.get_ylim()[0], 
                                xmin=0.0, xmax=1, color='black', lw=3.5)
       
        
    def get_hum_ratio(self, temperatures):
        """
        """
        hum_ratio = np.zeros(temperatures.shape, dtype=float)
        
        for key in self.point_dict:
            i = int(key.replace('_point', ''))
            hum_ratio[i] = self.point_dict[key][1][0]
    
        return hum_ratio
    
    def count_zones(self, temperatures, hum_ratios, humidities):
        """
        
            comfort line:
                    hum_ratio = -200/139 * temp + 6861/139
                    
                    
        """
        active_heat = len(temperatures[(temperatures  < 10)])
        
        passive_heat = len(temperatures[(temperatures >= 10) &
                                        (temperatures < 20)]
            )
        
        humidification = len(temperatures[(temperatures > 10.0) & 
                                          (humidities < 20) & 
                                          (hum_ratios < (-0.42042 * temperatures + 11.20840))])
        
                
        comfort = (humidities >= 20) & (humidities <= 80) & (temperatures > 20) & (temperatures < 26.66) & (hum_ratios <= ((-200/139) * temperatures + (6861/139)))
        
        ventilation = (humidities > 20) & (temperatures >= 20) & (temperatures <= 32.22) & (hum_ratios <= (-1.88853 * temperatures + 76.34873))
        
        evaporative = (temperatures >= 20.0) & (temperatures <= 40.55) & (humidities <= 80) & (hum_ratios >= (-0.42042 * temperatures + 11.20840)) & (hum_ratios <= (-0.40138 * temperatures + 24.58505)) & (hum_ratios <= (-1.39639 * temperatures + 62.72387))
        
        conditioning = (temperatures > 28.0) & (~comfort) & (~ventilation) & (~evaporative)
        
        evaporative = len(temperatures[evaporative & (~ventilation)])
        
        ventilation = len(temperatures[ventilation & (~comfort)])
        
        comfort = len(temperatures[comfort])
        
        conditioning = len(temperatures[conditioning])
        
#        print('active_heat', active_heat, '\n',
#              'passive', passive_heat, '\n',
#              'humidification', humidification,'\n',
#              'comfort', comfort,'\n',
#              'ventilation', ventilation,'\n',
#              'evaporative', evaporative, '\n',
#              'conditioning', conditioning
#              )
        
        total = active_heat + passive_heat + humidification + comfort + ventilation + evaporative + conditioning
        
        self.hist_vals = {'active_heat': active_heat / total,
                          'passive': passive_heat / total,
                          'humidification': humidification / total,
                          'comfort': comfort / total,
                          'ventilation': ventilation / total,
                          'evaporative': evaporative / total,
                          'conditioning': conditioning / total,
                          'total': total
                          }
        
        values = [active_heat / total,
                  passive_heat / total,
                  humidification / total,
                  comfort / total,
                  ventilation / total,
                  evaporative / total,
                  conditioning / total
                  ]
        columns = ['active_heat', 
                   'passive_heat',
                   'humidification',
                   'comfort',
                   'ventilation',
                   'evaporative',
                   'conditioning',
                   ]
        
        colors = [self.cmap(0.0),
                  self.cmap(0.15),
                  self.cmap(0.30),
                  self.cmap(0.45),
                  self.cmap(0.75),
                  self.cmap(0.90),
                  self.cmap(1.0)
                  ]
        
        import pandas as pd
        
        df = pd.DataFrame({'name': columns, 'values': values, 'cols': colors})
        
        self.zone_counts = df
        
    def make_points_dict(self, temperatures, humidities, weights):
        """
        """
        import copy
        point_marker = 'o'
        point_marker_size = 8
        point_alpha = 0.8
        
        """
            from pyscrochart import equations
            
            wet = equations.wet_bulb_temperature_empiric(dry_temp_c=np.array([10.0, 40.0]), relative_humid=np.array([0.20, 0.8]))
            hum_rat = equations.humidity_ratio_from_temps(dry_bulb_temp_c=37.5, wet_bulb_temp_c=wet) * 1000
            
            to get hum ratios which are the y values
        """
        max_temp = temperatures.max()
        min_temp = temperatures.min()
        max_weight = weights.max()
    
        point_template = {'label': 'data',
                          'style': {'color': None,
                                    'marker': point_marker, 
                                    'markersize': point_marker_size,
                                    'markeredgecolor': 'darkgrey',
                                    'alpha': point_alpha,
                                    'zorder': 1,
                                    'markeredgewidth': 0.4},
                          'xy': None
                          }
                          
        point_dict = {}
        
        for n, (temp, hums, weight) in enumerate(zip(temperatures, humidities, weights)):
            curr_point = copy.deepcopy(point_template)
            curr_point['xy'] = (temp, hums, weight)
            curr_point['style']['color'] = self.cmap(temp / (max_temp - min_temp))
            curr_point['style']['alpha'] = (weight / max_weight) * point_alpha
            
            point_dict['{}_point'.format(n)] = curr_point

            
        return point_dict
    
    def find_nearest(self, array, value):
        """
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
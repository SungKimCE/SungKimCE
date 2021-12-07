import os
import sys
import pandas as pd
import numpy as np

bom_directory = 'add_files/bom/'
rain_directory = 'rainfall_by_station_number'
high_temp_directory = 'highest_temp_by_station_number'
low_temp_directory = 'lowest_temp_by_station_number'

stns_file = 'add_files/stns/stn_nums_coords.csv'

def check_station_file(stn_number, directory): 
	for root, dirs, files in os.walk(directory): 
		for file in files: 
			if (str(stn_number) in file) and ('.zip' in file): 
				return 1 
	return 0 



df = pd.read_csv(stns_file)  

for index, row in df.iterrows():
	stn_number = int(row.stn_number)
	res1 = check_station_file(stn_number, os.path.join(bom_directory, high_temp_directory))
	res2 = check_station_file(stn_number, os.path.join(bom_directory, low_temp_directory)) 
	res3 = check_station_file(stn_number, os.path.join(bom_directory, rain_directory)) 
	if res1 * res2 * res3 == 1:
		df.loc[index,'ok'] = 1
	print(stn_number, res1 * res2 * res3)

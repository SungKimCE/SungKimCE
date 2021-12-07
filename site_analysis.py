import googlemaps
from datetime import datetime
import yaml
import os
import sys
import subprocess


##set empty yaml attributes to empty string instead of null
def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')


def set_sa_yaml(address, sa_folder, lat, lng):
    with open(r'site_analysis/settings.yaml') as file:
        order_a = yaml.load(file, Loader=yaml.FullLoader)

        
    yaml.add_representer(type(None), represent_none)

            
    order_a['site'] = address
    order_a['address'] = address
    order_a['lat'] = lat
    order_a['lng'] = lng
    order_a['order'] = sa_folder
    order_a['radius'] = 250/2
    yaml_file = 'site_analysis/settings.yaml'
            
    
    with open(yaml_file, 'w') as yaml_file:
        yaml.dump(order_a, yaml_file, default_flow_style=False)
    return yaml_file


def generate_siteanalysis():

	gmaps = googlemaps.Client(key = 'AIzaSyBRyIfcnH3PClFXVhZJlta9ctH_jujyfuI')
	sa_folder = sys.argv[1]
	address = sys.argv[2]

	#address = '1600 Amphitheatre Parkway, Mountain View, CA'
	geocode_result = gmaps.geocode(address)

	print(geocode_result[0]['geometry']['location'])

	lat = geocode_result[0]['geometry']['location']['lat']
	lng = geocode_result[0]['geometry']['location']['lng'] 

	print('lat: {}. lng: {}'.format(lat,lng))

	set_sa_yaml(address, sa_folder, lat, lng)    
	origWD = os.getcwd()
	os.makedirs('{}/SITE_ANALYSIS'.format(sa_folder))
	error = False
	try:
		#run site analysis script
		os.chdir(origWD + "/site_analysis/")
		run_sa = 'python3 main.py "' + sa_folder + '"'
		sa_status = subprocess.run(run_sa, shell = True)
		os.chdir(origWD)
		
		print(sa_status.check_returncode())
		order_status = "SA Done"
	except Exception as ex:
		print("SITE ANALYSIS ERROR")
		print(ex)

if __name__ == '__main__':
	generate_siteanalysis()
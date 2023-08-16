# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:37:30 2023

@author: descalante
"""

# Define the region of interest
filled_area_coords = [(-103.67, 31.58), (-103.655, 31.58), (-103.655, 31.635), (-103.67, 31.635)]



import requests
import pandas as pd
from io import StringIO

# URLs of the CSV files
#area_data_url = 'https://raw.githubusercontent.com/descalante-mhinvest/NOVILABS/main/DiamondBackGeometry.csv'
area_data_url = 'https://raw.githubusercontent.com/descalante-mhinvest/NOVILABS/main/mgy_areas.csv'
linestrings_url = 'https://raw.githubusercontent.com/descalante-mhinvest/NOVILABS/main/welldata.CSV'

# Download the area data CSV file
response = requests.get(area_data_url)
area_data = pd.read_csv(StringIO(response.text))

# Download the linestrings CSV file
response = requests.get(linestrings_url)
linestrings_data = pd.read_csv(StringIO(response.text))

from shapely import wkt

# Convert the string representations into actual Shapely geometries
#area_data['Geometry'] = area_data['Geometry'].apply(wkt.loads)
area_data['geometry'] = area_data['geometry'].apply(wkt.loads)
linestrings_data['geometry'] = linestrings_data['geometry'].apply(wkt.loads)

import geopandas as gpd
import matplotlib.pyplot as plt

# Convert the pandas dataframes into GeoPandas GeoDataFrames
area_gdf = gpd.GeoDataFrame(area_data, geometry='geometry')
linestrings_gdf = gpd.GeoDataFrame(linestrings_data, geometry='geometry')

from shapely.geometry import Point, Polygon, LineString

#plt.xlim([-97.25, -95.7])
#plt.ylim([29.7,30.8])
region_coords = [(-97.25, 29.7), (-95.7, 29.7), (-95.7, 30.8), (-97.25, 30.8)]

import numpy as np
region1 = Polygon(region_coords)

# Find the existing lines within this region
lines_in_region = linestrings_gdf[linestrings_gdf.geometry.intersects(region1)]
all_lines = linestrings_gdf.geometry
# Find the existing area within this region
region2 = area_gdf[area_gdf.geometry.intersects(region1)]
region3 = area_gdf.geometry

# Set the current coordinate reference system (CRS) to EPSG:4326 (WGS84)
lines_in_region.crs = "EPSG:4326"

# Transform the geometries to the UTM zone 14N (for Texas)
lines_in_region_meters = lines_in_region.to_crs("EPSG:32613")

# Calculate the length and direction of each line
lengths = lines_in_region_meters.geometry.length
# Find the median length and direction IN FEET
median_length_feet = np.median(lengths)

# Calculate the length and direction of each line
#lengths2 = lines_in_region.geometry.length
# Find the median length and direction IN LATT/LONG
#median_length = np.median(lengths2)
#median_direction = np.median(directions)

#median_length, median_direction

import numpy as np

# Define the region as the bounding box that contains the union polygon of the main island
region = region2.unary_union.envelope

region2_union1 = region2.unary_union

region2_union1_gdf = gpd.GeoDataFrame(geometry=[region2_union1])
region2_union1_gdf.crs = "EPSG:4326"

# Transform the geometries to the UTM zone 14N (for Texas)
region2_union1_feet = region2_union1_gdf.to_crs("EPSG:32613")

fig, ax = plt.subplots(figsize=(10, 10))
# Plot the old lines and the new lines
area_gdf.plot(ax=ax, color='blue')
#lines_in_region.plot(ax=ax, color='red')
#MGY AREA
plt.xlim([-97.25, -95.7])
plt.ylim([29.7,30.8])
plt.show()


from wktplot import WKTPlot

# Create a new WKTPlot
plot = WKTPlot(title="FANG MAP 1", save_dir="U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/")

# Add the area to the plot
for i in range(len(area_gdf)):
    plot.add_shape(area_gdf.geometry.iloc[i], fill_color="blue", fill_alpha=0.5)

# Add the original lines to the plot
for a in range(len(lines_in_region)):
    plot.add_shape(lines_in_region.geometry.iloc[a], line_color="red", line_alpha=0.5)


# Show the plot
plot.show()
# Plot each cluster with a different color
#plt.scatter(x_coords, y_coords, color=colors_list)

#grid.plot(ax=ax, color='black', markersize=10)
#exhib
#plt.xlim([-103.67, -103.655])
#plt.ylim([31.58,31.76])

#MGY AREA
plt.xlim([-97.25, -95.7])
plt.ylim([29.7,30.8])

#plt.savefig('U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/dbk_example_big')
#plt.savefig('U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/mgy_area_mike')
plt.show()

###############final calculations################ 

####transform units to meters and feet

#p = len(lines_in_region)
#q = np.average(lengths)
#total_old_well_length = p*q

area = region2_union1_feet.geometry.area
#total_density = ((total_old_well_length) // area)

area_value = area.values[0]  # This will get the first value from the series
print(f"{area_value:.1f}")
#print(total_old_well_length)
#print(total_density)


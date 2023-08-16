# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:03:41 2023

@author: descalante
"""

import requests
import pandas as pd
from io import StringIO
import numpy as np
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString

# URLs of the CSV files
area_data_url = 'https://raw.githubusercontent.com/descalante-mhinvest/NOVILABS/main/DiamondBackGeometry.csv'
area_data_url = 'https://github.com/descalante-mhinvest/NOVILABS/blob/main/mgy_areas.csv'
linestrings_url = 'https://raw.githubusercontent.com/descalante-mhinvest/NOVILABS/main/welldata.CSV'

# Download the area data CSV file
response = requests.get(area_data_url)
area_data = pd.read_csv(StringIO(response.text))

# Download the linestrings CSV file
response = requests.get(linestrings_url)
linestrings_data = pd.read_csv(StringIO(response.text))

# Convert the string representations into actual Shapely geometries
area_data['Geometry'] = area_data['Geometry'].apply(wkt.loads)
linestrings_data['geometry'] = linestrings_data['geometry'].apply(wkt.loads)

# Convert the pandas dataframes into GeoPandas GeoDataFrames
area_gdf = gpd.GeoDataFrame(area_data, geometry='Geometry')
linestrings_gdf = gpd.GeoDataFrame(linestrings_data, geometry='geometry')

'''
plt.xlim([-103.1, -103.35])
plt.ylim([31.3, 31.475])
'''
# Define the region of interest
#exhib4
#region_coords = [(-101.3, 31.6), (-101.8, 31.6), (-101.8, 31.9), (-101.3, 31.9)]
#exhib2
#region_coords = [(-103.1, 31.3), (-103.35, 31.3), (-103.35, 31.475), (-103.1, 31.475)]
#exhib3
#region_coords = [(-102.18, 31.45), (-102.55, 31.45), (-102.55, 31.95), (-102.18, 31.95)]
#echib1
#region_coords = [(-102.9, 31), (-103.4, 31), (-103.4, 31.3), (-102.9, 31.3)]
#EXHIB 5
region_coords = [(-103.35, 31.5), (-103.8, 31.5), (-103.8, 31.76), (-103.35, 31.76)]

region1 = Polygon(region_coords)

# Find the existing lines within this region IN LATT/LONG
lines_in_region = linestrings_gdf[linestrings_gdf.geometry.intersects(region1)]
# Find the existing area within this region IN LATT/LONG
region2 = area_gdf[area_gdf.geometry.intersects(region1)]

#######CHANGE COORDINATE SYSTEM
# Set the current coordinate reference system (CRS) to EPSG:4326 (WGS84)
lines_in_region.crs = "EPSG:4326"
# Transform the geometries to the UTM zone 14N (for Texas)
lines_in_region_meters = lines_in_region.to_crs("EPSG:32613")

# Calculate the length of each line IN METERS
lengths = lines_in_region_meters.geometry.length
# Find the median length IN METERS
median_length_meters = np.median(lengths)

# Calculate the length and direction of each line IN LATT/LONG
lengths2 = lines_in_region.geometry.length
# Find the median length and direction IN LATT/LONG
median_length = np.median(lengths2)
#median_direction = np.median(directions)

# Create a list to store the new lines
new_lines = []

# Define the size of each grid cell
grid_width = median_length / 13
grid_height = median_length

# Define the region as the bounding box that contains the union polygon of the main island
region2_union1 = region2.unary_union
buffer_distance = max(grid_width, grid_height)
region2_union = region2_union1.buffer(buffer_distance)
union_bounds = region2_union.bounds

# Transform the geometries to the UTM zone 14N (for Texas)
region2_union1_gdf = gpd.GeoDataFrame(geometry=[region2_union1])
region2_union1_gdf.crs = "EPSG:4326"
region2_union1_meters = region2_union1_gdf.to_crs("EPSG:32613")
'''
# Calculate the width and height of the polygon
width = 1.3*(bounds[2] - bounds[0]+grid_width)
height = 1*(bounds[3] - bounds[1]+grid_height)

# Calculate the center of the bounding box
#center_x = (bounds[0] + bounds[2]) / 2
#center_y = (bounds[1] + bounds[3]) / 2

# Calculate the number of steps in each direction
num_steps_x = int(width / grid_width)
num_steps_y = int(height / grid_height)
'''
                     ##########################begin clustering#########################

from sklearn.cluster import DBSCAN
from shapely.geometry import MultiLineString, MultiPoint, Polygon
from scipy.spatial import distance
from sklearn.cluster import MeanShift, estimate_bandwidth

# Calculate the direction of each line
directions = [np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]) for line in lines_in_region.geometry]

# Convert the directions to a 2D array with a single column
directions = np.array(directions).reshape(-1, 1)

# Estimate the bandwidth for the Mean Shift algorithm
bandwidth = estimate_bandwidth(directions)

# Fit the Mean Shift algorithm to the data
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(directions)

# Get the labels of the clusters
labels = ms.labels_

# Get the cluster centers (i.e., the mean direction of each cluster)
cluster_centers = ms.cluster_centers_

# Create an array of the start coordinates of each line
start_coords = np.vstack(lines_in_region.geometry.apply(lambda line: [line.xy[0][0], line.xy[1][0]]).to_numpy())

# Create an array of the end coordinates of each line
end_coords = np.vstack(lines_in_region.geometry.apply(lambda line: [line.xy[0][-1], line.xy[1][-1]]).to_numpy())

# Concatenate start and end coordinates
all_coords = np.concatenate((start_coords, end_coords), axis=0)
# Perform spatial clustering on the start coordinates
#db = DBSCAN(eps=0.3, min_samples=10).fit(start_coords)
db = DBSCAN(eps=0.1, min_samples=2).fit(all_coords)

# Create a DataFrame that associates each unique direction with a cluster label
#directions_df = pd.DataFrame({'direction': directions.squeeze(), 'cluster': db.labels_})

# Calculate the direction of each line in lines_in_region
lines_in_region.loc[:, 'direction'] = lines_in_region.geometry.apply(lambda line: np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]))

# Merge lines_in_region with directions_df to assign the cluster labels to the lines
#lines_df = pd.merge(lines_in_region, directions_df, on='direction', how='left')

# Calculate the orientation for each cluster
#orientations = lines_df.groupby('cluster').apply(lambda x: np.average([np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]) for line in x.geometry]))

for i in range(len(cluster_centers)):
    if cluster_centers[i] <0:
        cluster_centers[i] = cluster_centers[i] + np.pi
        
# Calculate the dominant direction
dominant_direction = np.mean(cluster_centers) - np.pi/2

# Initialize a dictionary to keep track of the number of lines in each cluster
lines_per_cluster = {}

# Create a convex hull around the start coordinates of the lines
hull = MultiPoint(all_coords).convex_hull

# Get the bounds of the convex hull
hull_bounds = hull.bounds

'''
# Calculate the center of the bounding box
center_x = (hull_bounds[0] + hull_bounds[2]) / 2
center_y = (hull_bounds[1] + hull_bounds[3]) / 2
'''
# Calculate
#the center of the bounding box
center_x = (union_bounds[0] + union_bounds[2]) / 2
center_y = (union_bounds[1] + union_bounds[3]) / 2

'''
# Calculate the width and height of the bounding box based on the convex hull
width = 1 * (hull_bounds[2] - hull_bounds[0])
height = 1 * (hull_bounds[3] - hull_bounds[1])
'''
# Calculate the width and height of the bounding box based on region2_union
width = 1 * (union_bounds[2] - union_bounds[0])
height = 1 * (union_bounds[3] - union_bounds[1])

# Calculate the number of steps in each direction
num_steps_x = int(width / grid_width)
num_steps_y = int(height / grid_height)

#grid of squares
grid_squares = []
grid_points = []

# Initialize a dictionary to keep track of the number of lines per cluster
lines_per_cluster = {}

# Separate the lines by cluster
lines_by_cluster = [lines_in_region[labels == label] for label in np.unique(labels)]

# Initialize a list to store the new lines
new_lines = []
x_coords = []
y_coords = []
colors_list = []

# Define your colors
colors = ['green', 'purple', 'orange', 'cyan']
# Create a mapping from cluster labels to colors
color_map = {label: color for label, color in zip(np.unique(labels), colors)}


# For each cluster of lines...
for cluster_label, lines_in_cluster in enumerate(lines_by_cluster):
    '''
    # Calculate the bounding box of the lines in the cluster
    min_x = lines_in_cluster.geometry.bounds.minx.min()
    max_x = lines_in_cluster.geometry.bounds.maxx.max()
    min_y = lines_in_cluster.geometry.bounds.miny.min()
    max_y = lines_in_cluster.geometry.bounds.maxy.max()
    
    # Calculate the number of steps in each direction
    num_steps_x = int((max_x - min_x) / grid_width)
    num_steps_y = int((max_y - min_y) / grid_height)
    '''
    
    # Get the dominant direction for the cluster from cluster_centers
    dominant_direction = cluster_centers[cluster_label][0] - np.pi/2
    '''
    directions = [np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]) for line in lines_in_cluster.geometry]
    #dominant_direction = np.median(directions) - np.pi/2
    
    #Sort the directions
    directions.sort()
    
    # Calculate indices for 10th and 90th percentile
    lower_index = int(len(directions) * 0.10)
    upper_index = int(len(directions) * 0.90)
    
    # Filter directions to take only the middle 80%
    filtered_directions = directions[lower_index:upper_index]
    
    # Calculate dominant direction
    dominant_direction = np.median(filtered_directions) - np.pi/2
    '''
    # Create the rotation matrix for the cluster
    rotation_matrix = np.array([[np.cos(dominant_direction), -np.sin(dominant_direction)],
                                [np.sin(dominant_direction), np.cos(dominant_direction)]])
    
    # Generate new lines based on the cluster orientations
    for i in range(num_steps_x):
        for j in range(num_steps_y):
            
            # Calculate the x and y coordinates as offsets from the center point
            x = (i - num_steps_x / 2) * grid_width
            y = (j - num_steps_y / 2) * grid_height
            '''
            # Calculate the distance from the new point to the start and end points of the lines
            distances = distance.cdist([[x, y]], start_coords, 'euclidean')
            # Find the index of the closest line
            closest_index = np.argmin(distances)
            # Find the index of the closest liner
            cluster_label = labels[closest_index]
            
            # Use the orientation of the cluster to generate the new line
            
            rotation_angle = cluster_centers[labels[closest_index]][0]
                      
        
            # Create the rotation matrix
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                        [np.sin(rotation_angle), np.cos(rotation_angle)]])
            
            '''
            # Rotate the coordinates
            x_rotated, y_rotated = rotation_matrix @ np.array([x, y])
    
            # Shift the coordinates back
            x_rotated += center_x
            y_rotated += center_y
    
            # Calculate the distance from the new point to the start and end points of the lines
            distances = distance.cdist([[x_rotated, y_rotated]], start_coords, 'euclidean')
            # Find the index of the closest line
            closest_index = np.argmin(distances)
            
            # Get the label of the cluster that the rectangle belongs to
            cluster_label = labels[closest_index]
            
            # Use the orientation of the cluster to generate the new line
            rotation_angle = cluster_centers[labels[closest_index]]
            
            # use the orientation of the closest line to generate the new line
            rotation_angle = directions[closest_index]
    
            # Calculate the end point of the line based on the rotation angle
            x_end = x_rotated + median_length * np.cos(rotation_angle)
            y_end = y_rotated + median_length * np.sin(rotation_angle)
            y_end2 = y_rotated + median_length * np.sin(rotation_angle+np.pi)
            x_end2 = x_rotated + median_length * np.cos(rotation_angle+np.pi)
            x_end3 = x_rotated + 2*median_length * np.cos(rotation_angle)
            y_end3 = y_rotated + 2*median_length * np.sin(rotation_angle)
            
            new_line = LineString([(x_rotated, y_rotated), (x_end, y_end)])
            x_coords.append(x_rotated)
            y_coords.append(y_rotated)
            #colors_list.append(color_map[cluster_label])
            # Check if the new line intersects the union
            if new_line.intersects(region2_union1):
                # If the new line intersects the union, keep only the part of the line that is within the union
                new_line_in_union = new_line.intersection(region2_union1)
                
                # Create a new square with the median length and direction
                #new_square = Polygon([(x_rotated-grid_width/2, y_rotated), ((x_rotated+grid_width/2), y_rotated), ((x_end+grid_width/2), y_end), (x_end-grid_width/2, y_end), (x_rotated-grid_width/2, y_rotated)])
                #plus_square = Polygon([(x_rotated-grid_width/2, y_rotated), ((x_rotated+grid_width/2), y_rotated), ((x_end+grid_width/2), y_end2), (x_end-grid_width/2, y_end2), (x_rotated-grid_width/2, y_rotated)])
                #from shapely.affinity import rotate
    
                # Calculate the points for the new square
                p1 = (x_rotated + grid_width/2 * np.cos(rotation_angle - np.pi/2), y_rotated + grid_width/2 * np.sin(rotation_angle - np.pi/2))
                p2 = (x_rotated + grid_width/2 * np.cos(rotation_angle + np.pi/2), y_rotated + grid_width/2 * np.sin(rotation_angle + np.pi/2))
                p3 = (x_end + grid_width/2 * np.cos(rotation_angle + np.pi/2), y_end + grid_width/2 * np.sin(rotation_angle + np.pi/2))
                p4 = (x_end + grid_width/2 * np.cos(rotation_angle - np.pi/2), y_end + grid_width/2 * np.sin(rotation_angle - np.pi/2))
    
                # Create the new square
                new_square = Polygon([p1, p2, p3, p4])
                
                # Calculate the points for the new square
                p1 = (x_rotated + grid_width/2 * np.cos(rotation_angle - np.pi/2), y_rotated + grid_width/2 * np.sin(rotation_angle - np.pi/2))
                p2 = (x_rotated + grid_width/2 * np.cos(rotation_angle + np.pi/2), y_rotated + grid_width/2 * np.sin(rotation_angle + np.pi/2))
                p3 = (x_end2 + grid_width/2 * np.cos(rotation_angle + np.pi/2), y_end2 + grid_width/2 * np.sin(rotation_angle + np.pi/2))
                p4 = (x_end2 + grid_width/2 * np.cos(rotation_angle - np.pi/2), y_end2 + grid_width/2 * np.sin(rotation_angle - np.pi/2))
    
                # Create the new square
                new_square2 = Polygon([p1, p2, p3, p4])
                
                # Calculate the points for the new square
                p1 = (x_end + grid_width/2 * np.cos(rotation_angle + np.pi/2), y_end + grid_width/2 * np.sin(rotation_angle + np.pi/2))
                p2 = (x_end + grid_width/2 * np.cos(rotation_angle - np.pi/2), y_end + grid_width/2 * np.sin(rotation_angle - np.pi/2))
                p3 = (x_end3 + grid_width/2 * np.cos(rotation_angle - np.pi/2), y_end3 + grid_width/2 * np.sin(rotation_angle - np.pi/2))
                p4 = (x_end3 + grid_width/2 * np.cos(rotation_angle + np.pi/2), y_end3 + grid_width/2 * np.sin(rotation_angle + np.pi/2))
    
                # Create the new square
                new_square3 = Polygon([p1, p2, p3, p4])
                
                # Add the new square to the list of grid squares
                grid_squares.append(new_square)
                grid_squares.append(new_square2)
                grid_squares.append(new_square3)
                # Check if the new line intersects any existing lines
                intersects_existing_line = any(new_line_in_union.intersects(line) for line in lines_in_region.geometry)
                if not intersects_existing_line:
                    
                    '''
                    # Define the buffer distance
                    buffer_distance = grid_width * .05
                    
                    # Initialize a list to store the buffered new lines
                    buffered_new_lines = []
                    
                    for line in new_lines:
                        # Create a buffer around the new line
                        buffered_new_line = new_line_in_union.buffer(buffer_distance, cap_style=2)
                        # Add the buffered new line to the list
                        buffered_new_lines.append(buffered_new_line)
                    # Convert the list of buffered new lines into a GeoSeries
                    buffered_new_lines_gdf = gpd.GeoSeries(buffered_new_lines)
                    
                    # Combine all the buffered new lines into a single shape
                    buffered_new_lines_union = buffered_new_lines_gdf.unary_union
                    intersects_buffered_new_lines = new_line_in_union.intersects(buffered_new_lines_union)
                    '''
                    # Check if the new line intersects any new lines
                    new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)
                    intersects_new_line = any(new_line_in_union.intersects(line) for line in new_lines_gdf.geometry)
                    if not intersects_new_line:
                        '''
                        new_lines.append(new_line_in_union)
                        '''
                        
                        '''
                        intersects_buffered_new_lines = new_line_in_union.intersects(buffered_new_lines_union)
                        if intersects_buffered_new_lines:
                            # If the new line intersects the buffered new lines, calculate the length of the intersection
                            intersection = new_line_in_union.intersection(buffered_new_lines_union)
                            intersection_length = intersection.length
                        
                            # Check if the length of the intersection is less than a third of the length of the new line
                            if intersection_length < median_length * 1 :
                                # If the new line does not intersect any existing lines, does not intersect any new lines, and is not too close to any new lines, add it to the list of new lines
                                new_lines.append(new_line_in_union)
                        else:
                            new_lines.append(new_line_in_union)
                        
                        '''
                        
                        # Calculate the area of the intersection with blue area
                        intersection_area = region2_union1.intersection(new_square).area
                        #convert to length
                        intersection_area_length = np.sqrt(intersection_area)
                        
                        # Calculate the density
                        total_intersection_length = 0
                        # Initialize a counter for the number of lines in this rectangle
                        lines_in_this_square = 0
                        new_lines_in_this_square = 0 
                        
                        
                        for line in new_lines:
                            # Check if the rectangle intersects with the line and if the intersection is a LineString
                            if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                                # Calculate the length of the intersection
                                intersection_length = new_square.intersection(line).length
    
                                # Add the length of the intersection to the total
                                total_intersection_length += intersection_length
                                new_lines_in_this_square += 1
                                
                        
                        for line in lines_in_region.geometry:  
                            # Check if the rectangle intersects with the line and if the intersection is a LineString
                            if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                                # Increment the counter
                                lines_in_this_square += 1
                                
                        # Check if the total length of intersections is less than half of the rectangle's perimeter
                        if total_intersection_length / intersection_area < .60 and lines_in_this_square<1 and new_lines_in_this_square<2:
                            #grid_points.append((x_rotated, y_rotated))
                            new_lines.append(new_line_in_union)
                        

# Convert the list of grid cells into a GeoDataFrame
grid = gpd.GeoDataFrame(geometry=grid_squares)

'''
# Iterate over each rectangle in the grid
for rectangle in grid.geometry:
    
    # Calculate the area of the intersection with blue area
    intersection_area = region2_union1.intersection(rectangle).area
        
    # Calculate the center of the rectangle
    center_x = rectangle.bounds[0] + grid_width/2 
    center_y = rectangle.bounds[1] + grid_height /2

    # Calculate the distance from the center of the rectangle to the start and end points of the lines
    distances = distance.cdist([[center_x, center_y]], start_coords, 'euclidean')

    # Find the index of the closest line
    closest_index = np.argmin(distances)
        
    # Use the orientation of the cluster to generate the new line
    rotation_angle = cluster_centers[labels[closest_index]]

    # Calculate the start and end points of the line based on the rotation angle
    start_x = center_x - grid_height / 2 * np.cos(rotation_angle)
    start_y = center_y - grid_height / 2 * np.sin(rotation_angle)
        
    end_x = center_x + grid_height / 2 * np.cos(rotation_angle)
    end_y = center_y + grid_height / 2 * np.sin(rotation_angle)
            
    # Create the new line
    new_line = LineString([(start_x, start_y), (end_x, end_y)])
    
    # Calculate the points for the new square
    p1 = (start_x + grid_width/2 * np.cos(rotation_angle - np.pi/2), start_y + grid_width/2 * np.sin(rotation_angle - np.pi/2))
    p2 = (start_x + grid_width/2 * np.cos(rotation_angle + np.pi/2), start_y + grid_width/2 * np.sin(rotation_angle + np.pi/2))
    p3 = (end_x + grid_width/2 * np.cos(rotation_angle + np.pi/2), end_y + grid_width/2 * np.sin(rotation_angle + np.pi/2))
    p4 = (end_x + grid_width/2 * np.cos(rotation_angle - np.pi/2), end_y + grid_width/2 * np.sin(rotation_angle - np.pi/2))

    # Create the new square
    new_square = Polygon([p1, p2, p3, p4])
    if new_line.intersects(region2_union1):
        # If the new line intersects the union, keep only the part of the line that is within the union
        new_line_in_union = new_line.intersection(region2_union1)
        intersects_existing_line = any(new_line_in_union.intersects(line) for line in lines_in_region.geometry)
        if not intersects_existing_line:
            
            # Check if the new line intersects any new lines
            new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)
            intersects_new_line = any(new_line_in_union.intersects(line) for line in new_lines_gdf.geometry)
            if not intersects_new_line:
                # Calculate the area of the intersection with blue area
                intersection_area = region2_union1.intersection(new_square).area
                #convert to length
                intersection_area_length = np.sqrt(intersection_area)
                # Calculate the density
                total_intersection_length = 0
                # Initialize a counter for the number of lines in this rectangle
                lines_in_this_square = 0
                new_lines_in_this_square = 0 
                
                for line in new_lines:
                    # Check if the rectangle intersects with the line and if the intersection is a LineString
                    if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                        # Calculate the length of the intersection
                        intersection_length = new_square.intersection(line).length
    
                        # Add the length of the intersection to the total
                        total_intersection_length += intersection_length
                        new_lines_in_this_square += 1
                        
                for line in lines_in_region.geometry:  
                    # Check if the rectangle intersects with the line and if the intersection is a LineString
                    if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                        # Increment the counter
                        lines_in_this_square += 1
                        
                # Check if the total length of intersections is less than half of the rectangle's perimeter
                if total_intersection_length / intersection_area < .75 and lines_in_this_square<1 and new_lines_in_this_square<1:
                    #grid_points.append((x_rotated, y_rotated))
                    new_lines.append(new_line_in_union)
                
    # Calculate the lower left corner of the rectangle
    lower_left_x = rectangle.bounds[0]
    lower_left_y = rectangle.bounds[1]

    # Calculate the upper left corner of the rectangle
    upper_left_x = lower_left_x + median_length * np.cos(rotation_angle)
    upper_left_y = lower_left_y + median_length * np.sin(rotation_angle)

    # Calculate the end points of the line based on the rotation angle
    start_x = lower_left_x
    start_y = lower_left_y
    end_x = upper_left_x
    end_y = upper_left_y

    #Create the new line
    new_line = LineString([(start_x, start_y), (end_x, end_y)])
    
    # Calculate the points for the new square
    p1 = (start_x + grid_width/2 * np.cos(rotation_angle - np.pi/2), start_y + grid_width/2 * np.sin(rotation_angle - np.pi/2))
    p2 = (start_x + grid_width/2 * np.cos(rotation_angle + np.pi/2), start_y + grid_width/2 * np.sin(rotation_angle + np.pi/2))
    p3 = (end_x + grid_width/2 * np.cos(rotation_angle + np.pi/2), end_y + grid_width/2 * np.sin(rotation_angle + np.pi/2))
    p4 = (end_x + grid_width/2 * np.cos(rotation_angle - np.pi/2), end_y + grid_width/2 * np.sin(rotation_angle - np.pi/2))

    # Create the new square
    new_square = Polygon([p1, p2, p3, p4])
    
    if new_line.intersects(region2_union1):
        # If the new line intersects the union, keep only the part of the line that is within the union
        new_line_in_union = new_line.intersection(region2_union1)
        intersects_existing_line = any(new_line_in_union.intersects(line) for line in lines_in_region.geometry)
        if not intersects_existing_line:
            
            # Check if the new line intersects any new lines
            new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)
            intersects_new_line = any(new_line_in_union.intersects(line) for line in new_lines_gdf.geometry)
            if not intersects_new_line:
                # Calculate the area of the intersection with blue area
                intersection_area = region2_union1.intersection(new_square).area
                #convert to length
                intersection_area_length = np.sqrt(intersection_area)
                # Calculate the density
                total_intersection_length = 0
                # Initialize a counter for the number of lines in this rectangle
                lines_in_this_square = 0
                new_lines_in_this_square = 0 
                
                for line in new_lines:
                    # Check if the rectangle intersects with the line and if the intersection is a LineString
                    if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                        # Calculate the length of the intersection
                        intersection_length = new_square.intersection(line).length
    
                        # Add the length of the intersection to the total
                        total_intersection_length += intersection_length
                        new_lines_in_this_square += 1
                        
                for line in lines_in_region.geometry:  
                    # Check if the rectangle intersects with the line and if the intersection is a LineString
                    if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                        # Increment the counter
                        lines_in_this_square += 1
                        
                # Check if the total length of intersections is less than half of the rectangle's perimeter
                if total_intersection_length / intersection_area < .75 and lines_in_this_square<1 and new_lines_in_this_square<1:
                    #grid_points.append((x_rotated, y_rotated))
                    new_lines.append(new_line_in_union)
            
    # Calculate the lower right corner of the rectangle
    lower_right_x = rectangle.bounds[2]
    lower_right_y = rectangle.bounds[3]

    # Calculate the upper right corner of the rectangle
    # Calculate the end point of the line based on the rotation angle
    upper_right_x = lower_right_x - median_length * np.cos(rotation_angle)
    upper_right_y = lower_right_y - median_length * np.sin(rotation_angle)

    # Calculate the end points of the line based on the rotation angle
    start_x = lower_right_x
    start_y = lower_right_y
    end_x = upper_right_x
    end_y = upper_right_y

    # Create the new line
    new_line = LineString([(start_x, start_y), (end_x, end_y)])
    
    # Calculate the points for the new square
    p1 = (start_x + grid_width/2 * np.cos(rotation_angle - np.pi/2), start_y + grid_width/2 * np.sin(rotation_angle - np.pi/2))
    p2 = (start_x + grid_width/2 * np.cos(rotation_angle + np.pi/2), start_y + grid_width/2 * np.sin(rotation_angle + np.pi/2))
    p3 = (end_x + grid_width/2 * np.cos(rotation_angle + np.pi/2), end_y + grid_width/2 * np.sin(rotation_angle + np.pi/2))
    p4 = (end_x + grid_width/2 * np.cos(rotation_angle - np.pi/2), end_y + grid_width/2 * np.sin(rotation_angle - np.pi/2))

    # Create the new square
    new_square = Polygon([p1, p2, p3, p4])
    if new_line.intersects(region2_union1):
        # If the new line intersects the union, keep only the part of the line that is within the union
        new_line_in_union = new_line.intersection(region2_union1)
        intersects_existing_line = any(new_line_in_union.intersects(line) for line in lines_in_region.geometry)
        if not intersects_existing_line:
            
            # Check if the new line intersects any new lines
            new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)
            intersects_new_line = any(new_line_in_union.intersects(line) for line in new_lines_gdf.geometry)
            if not intersects_new_line:
                # Calculate the area of the intersection with blue area
                intersection_area = region2_union1.intersection(new_square).area
                #convert to length
                intersection_area_length = np.sqrt(intersection_area)
                # Calculate the density
                total_intersection_length = 0
                # Initialize a counter for the number of lines in this rectangle
                lines_in_this_square = 0
                new_lines_in_this_square = 0 
                
                for line in new_lines:
                    # Check if the rectangle intersects with the line and if the intersection is a LineString
                    if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                        # Calculate the length of the intersection
                        intersection_length = new_square.intersection(line).length
    
                        # Add the length of the intersection to the total
                        total_intersection_length += intersection_length
                        new_lines_in_this_square += 1
                        
                for line in lines_in_region.geometry:  
                    # Check if the rectangle intersects with the line and if the intersection is a LineString
                    if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                        # Increment the counter
                        lines_in_this_square += 1
                        
                # Check if the total length of intersections is less than half of the rectangle's perimeter
                if total_intersection_length / intersection_area < .75 and lines_in_this_square<1 and new_lines_in_this_square<1:
                    #grid_points.append((x_rotated, y_rotated))
                    new_lines.append(new_line_in_union)

'''

        
                                            #end clustering 

# Convert the list of new lines into a GeoDataFrame
new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)


fig, ax = plt.subplots(figsize=(10, 10))
# Plot the old lines and the new lines
area_gdf.plot(ax=ax, color='blue')
#grid.plot(ax=ax, color='black')
lines_in_region.plot(ax=ax, color='red')
#new_lines_gdf.plot(ax=ax, color='yellow')

from wktplot import WKTPlot

# Create a new WKTPlot
plot = WKTPlot(title="FANG MAP 1", save_dir="U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/")

# Add the area to the plot
for i in range(len(area_gdf)):
    plot.add_shape(area_gdf.geometry.iloc[i], fill_color="blue", fill_alpha=0.5)

# Add the original lines to the plot
for a in range(len(lines_in_region)):
    plot.add_shape(lines_in_region.geometry.iloc[a], line_color="red", line_alpha=0.5)

# Add the new lines to the plot
for a in range(len(new_lines)):
    plot.add_shape(new_lines[a], line_color="yellow", line_alpha=0.5)

# Show the plot
plot.show()
# Plot each cluster with a different color
#plt.scatter(x_coords, y_coords, color=colors_list)

#grid.plot(ax=ax, color='black', markersize=10)


#exhib3
#plt.xlim([-102.55, -102.18])
#plt.ylim([31.45, 31.95])

#exhib2
#plt.xlim([-103.35, -103.1])
#plt.ylim([31.3, 31.475])

#exhib1
#plt.xlim([-103.4, -102.9])
#plt.ylim([31,31.3])

#exhib4
#plt.xlim([-101.8, -101.3])
#plt.ylim([31.6,31.9])

#exhib5
plt.xlim([-103.8, -103.35])
plt.ylim([31.5,31.76])

plt.savefig('U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/dbk_example_big')
plt.show()

###############final calculations################ 

####transform units to meters and feet

# Set the current coordinate reference system (CRS) to EPSG:4326 (WGS84)
new_lines_gdf.crs = "EPSG:4326"

# Transform the geometries to the UTM zone 14N (for Texas)
new_lines_gdf_meters = new_lines_gdf.to_crs("EPSG:32613")

lengths2 = new_lines_gdf_meters.geometry.length
    
t = len(new_lines)
y = np.average(lengths2)
p = len(lines_in_region)
q = np.average(lengths)
total_new_well_length = y*t
total_old_well_length = p*q

# number of net new lines 
num_new_lines = total_new_well_length / median_length_meters

area = region2_union1_meters.geometry.area
total_density = ((total_new_well_length + total_old_well_length) // area)

print(total_new_well_length)
print(total_old_well_length)
print(num_new_lines)
area_value = area.values[0]  # This will get the first value from the series
print(f"{area_value:.1f}")
print(median_length_meters)
print(total_density)


#if total_intersection_length / intersection_area < .95 and lines_in_this_square<1 and new_lines_in_this_square<2:



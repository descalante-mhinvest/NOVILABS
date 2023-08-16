# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 02:31:26 2023

@author: descalante
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:03:41 2023

@author: descalante
"""

import requests
import pandas as pd
from io import StringIO
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiLineString, MultiPoint, Polygon
from scipy.spatial import distance
from sklearn.cluster import MeanShift, estimate_bandwidth
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import box
import numpy as np

from geopandas.tools import sjoin
from shapely.geometry import Point, Polygon

# URLs of the CSV files
area_data_url = 'https://raw.githubusercontent.com/descalante-mhinvest/NOVILABS/main/DiamondBackGeometry.csv'
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

# Define the coordinates of the completely filled area in the correct order
filled_area_coords = [(-102.235, 32.45), (-102.18, 32.45), (-102.18, 32.36), (-102.235, 32.36)]

# Create a Shapely polygon from these coordinates
filled_area = Polygon(filled_area_coords)

# Calculate the number of wells in this area
num_wells = linestrings_gdf[linestrings_gdf.geometry.intersects(filled_area)].shape[0]

# Calculate the area
area = filled_area.area

# Calculate the density
density = num_wells / area



# Create a grid over the entire area
x_min, y_min, x_max, y_max = area_gdf.total_bounds  # replace 'gdf' with your GeoDataFrame

grid_width = 0.3  # set grid width
grid_height = 0.3  # set grid height

grid_points = []
for x in np.arange(x_min, x_max, grid_width):
    for y in np.arange(y_min, y_max, grid_height):
        grid_points.append(Point(x, y))


grid_polygons = [Polygon([(p.x - grid_width / 2, p.y - grid_height / 2),
                          (p.x + grid_width / 2, p.y - grid_height / 2),
                          (p.x + grid_width / 2, p.y + grid_height / 2),
                          (p.x - grid_width / 2, p.y + grid_height / 2)])
                 for p in grid_points]

grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons)

# Perform a spatial join between the grid and the blue areas
joined_gdf = sjoin(grid_gdf, area_gdf, how='inner', op='intersects')

# Drop duplicates based on the 'geometry' column
joined_gdf = joined_gdf.drop_duplicates(subset='geometry')

                                     #######start iteration
        
# Define the region of interest
#region_coords = [(-102.9, 31.3), (-103.4, 31.3), (-103.4, 31.5), (-102.9, 31.5)]
#region_coords = [(-102.9, 31), (-103.4, 31), (-103.4, 31.29), (-102.9, 31.29)]
#region1 = Polygon(region_coords)

number_of_new_lines = []

for index, row in joined_gdf.iterrows():
    # Get the geometry of the row, which is a Polygon
    region1 = row.geometry
    
    union_bounds = region1.bounds
    # Find the existing lines within this region
    lines_in_region = linestrings_gdf[linestrings_gdf.geometry.intersects(region1)]
        # If there are no lines in the region, skip the main code and just plot the existing lines and area
    if len(lines_in_region) == 0:
        # Plot the existing lines and area
        fig, ax = plt.subplots()
        area_gdf.plot(ax=ax, color='blue')
        plt.xlim([region1.bounds[0], region1.bounds[2]])
        plt.ylim([region1.bounds[1], region1.bounds[3]])
        plt.savefig(f'U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/dbk_example_nolines_{index}')
        plt.show()
    else:
        all_lines = linestrings_gdf.geometry
        # Find the existing area within this region
        region2 = area_gdf[area_gdf.geometry.intersects(region1)]
        region3 = area_gdf.geometry
        
        
        # Calculate the length and direction of each line
        lengths = lines_in_region.geometry.length
        
        # Find the median length and direction
        median_length = np.median(lengths)
        #median_direction = np.median(directions)
        
        #median_length, median_direction
        
        
    
        
        # Create a list to store the new lines
        new_lines = []
        
        # Define the size of each grid cell
        grid_width = median_length / 13
        grid_height = median_length
        
        # Calculate the rotation angle to align the grid with the direction of the lines
        #rotation_angle = np.average(directions_no_outliers)
        #rotation_angle_perp = (np.pi/2)+rotation_angle
        
        # Create a list of rotation angles
        #rotation_angles = [rotation_angle, rotation_angle_perp]
        
        # Define the region as the bounding box that contains the union polygon of the main island
        region = region2.unary_union.envelope
        
        
        region2_union1 = region2.unary_union
        
        buffer_distance = max(grid_width, grid_height)
        region2_union = region2_union1.buffer(buffer_distance)
        
        bounds = region2_union.bounds
        
        # Calculate the width and height of the polygon
        #width = 1.3*(bounds[2] - bounds[0]+grid_width)
        #height = 1*(bounds[3] - bounds[1]+grid_height)
        
        # Calculate the number of steps in each direction
        #num_steps_x = int(width / grid_width)
        #num_steps_y = int(height / grid_height)
        
                                                #begin clustering
        
        
        
        # Calculate the direction of each line
        directions = [np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]) for line in lines_in_region.geometry]
        
        # Convert the directions to a 2D array with a single column
        directions = np.array(directions).reshape(-1, 1)
        
        # Estimate the bandwidth for the Mean Shift algorithm
        bandwidth = estimate_bandwidth(directions)
        
        if bandwidth > 0:
            
            # Fit the Mean Shift algorithm to the data
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(directions)
            
            # Get the labels of the clusters
            labels = ms.labels_
            
            # Get the cluster centers (i.e., the mean direction of each cluster)
            cluster_centers = ms.cluster_centers_
            
            # Create an array of the start coordinates of each line
            start_coords = np.vstack(lines_in_region.geometry.apply(lambda line: [line.xy[0][0], line.xy[1][0]]).to_numpy())
            
            # Perform spatial clustering on the start coordinates
            #db = DBSCAN(eps=0.3, min_samples=10).fit(start_coords)
            db = DBSCAN(eps=0.3, min_samples=10).fit(start_coords)
            
            
            # Create a DataFrame that associates each unique direction with a cluster label
            directions_df = pd.DataFrame({'direction': directions.squeeze(), 'cluster': db.labels_})
            
            # Calculate the direction of each line in lines_in_region
            lines_in_region.loc[:, 'direction'] = lines_in_region.geometry.apply(lambda line: np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]))
            
            # Merge lines_in_region with directions_df to assign the cluster labels to the lines
            lines_df = pd.merge(lines_in_region, directions_df, on='direction', how='left')
            
            # Calculate the orientation for each cluster
            orientations = lines_df.groupby('cluster').apply(lambda x: np.average([np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]) for line in x.geometry]))
            
            # Calculate the center of the bounding box
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            
            # Calculate the center of the bounding box
            center_x = (union_bounds[0] + union_bounds[2]) / 2
            center_y = (union_bounds[1] + union_bounds[3]) / 2
            
            # Calculate the dominant direction
            dominant_direction = np.mean(cluster_centers)
            
            # Create the rotation matrix
            rotation_matrix = np.array([[np.cos(dominant_direction), -np.sin(dominant_direction)],
                                        [np.sin(dominant_direction), np.cos(dominant_direction)]])
            
            # Initialize a dictionary to keep track of the number of lines in each cluster
            lines_per_cluster = {}
            
            
            
            # Create a convex hull around the start coordinates of the lines
            hull = MultiPoint(start_coords).convex_hull
            
            # Get the bounds of the convex hull
            hull_bounds = hull.bounds
            
            # Calculate the center of the bounding box
            #center_x = (hull_bounds[0] + hull_bounds[2]) / 2
            #center_y = (hull_bounds[1] + hull_bounds[3]) / 2
            
            # Calculate the width and height of the bounding box based on the convex hull
            width = 1 * (hull_bounds[2] - hull_bounds[0])
            height = 1 * (hull_bounds[3] - hull_bounds[1])
            
            # Calculate the number of steps in each direction
            num_steps_x = int(width / grid_width)
            num_steps_y = int(height / grid_height)+3
            
            #grid of squares
            grid_squares = []
            grid_points = []
            
            # Initialize a dictionary to keep track of the number of lines per cluster
            lines_per_cluster = {}
            
            # Calculate the width and height of the bounding box based on region2_union
            width = 1 * (union_bounds[2] - union_bounds[0])
            height = 1 * (union_bounds[3] - union_bounds[1])

            # Calculate the number of steps in each direction
            num_steps_x = int(width / grid_width)
            num_steps_y = int(height / grid_height)
            
                        
            # Separate the lines by cluster
            lines_by_cluster = [lines_in_region[labels == label] for label in np.unique(labels)]
                        
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
                dominant_direction = cluster_centers[cluster_label][0]
                '''
                directions = [np.arctan2(line.xy[1][1] - line.xy[1][0], line.xy[0][1] - line.xy[0][0]) for line in lines_in_cluster.geometry]
                dominant_direction = np.median(directions) - np.pi/2
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
                
                        # Calculate the end point of the line based on the rotation angle
                        x_end = x_rotated + median_length * np.cos(rotation_angle)
                        y_end = y_rotated + median_length * np.sin(rotation_angle)
                        y_end2 = y_rotated + median_length * np.sin(rotation_angle+np.pi)
                        x_end2 = x_rotated + median_length * np.cos(rotation_angle+np.pi)
                        x_end3 = x_rotated + 2*median_length * np.cos(rotation_angle)
                        y_end3 = y_rotated + 2*median_length * np.sin(rotation_angle)
                        
                        new_line = LineString([(x_rotated, y_rotated), (x_end, y_end)])
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
                                    
                                    for line in new_lines:
                                        # Check if the rectangle intersects with the line and if the intersection is a LineString
                                        if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                                            # Calculate the length of the intersection
                                            intersection_length = new_square.intersection(line).length
                
                                            # Add the length of the intersection to the total
                                            total_intersection_length += intersection_length
                                    for line in lines_in_region.geometry:  
                                        # Check if the rectangle intersects with the line and if the intersection is a LineString
                                        if new_square.intersects(line) and isinstance(new_square.intersection(line), LineString):
                                            # Increment the counter
                                            lines_in_this_square += 1
                                            
                                    # Check if the total length of intersections is less than half of the rectangle's perimeter
                                    if total_intersection_length < intersection_area_length*1.8 and lines_in_this_square<10:
                                        #grid_points.append((x_rotated, y_rotated))
                                        
                                        # If the new line does not intersect any existing lines, add it to the list of new lines
                                        new_lines.append(new_line_in_union)
                
                        
                    
            # Convert the list of grid cells into a GeoDataFrame
            grid = gpd.GeoDataFrame(geometry=grid_squares)
            
            
            
            # Iterate over each rectangle in the grid
            for rectangle in grid.geometry:
                # Initialize a variable to keep track of the total length of intersections
                total_intersection_length = 0
                
                # Calculate the area of the intersection with blue area
                intersection_area = region2_union1.intersection(rectangle).area
                #convert to length
                intersection_area_length = np.sqrt(intersection_area)
            
                # Iterate over each new line
                for line in new_lines:
                    # Check if the rectangle intersects with the line and if the intersection is a LineString
                    if rectangle.intersects(line) and isinstance(rectangle.intersection(line), LineString):
                        # Calculate the length of the intersection
                        intersection_length = rectangle.intersection(line).length
            
                        # Add the length of the intersection to the total
                        total_intersection_length += intersection_length
                        
                        # Increment the counter
                        lines_in_this_square += 1
                    
                    
                # Check if the total length of intersections is less than half of the rectangle's perimeter
                if total_intersection_length < intersection_area_length / 5: #and lines_in_this_square < 100:
                    ########################## The rectangle is in an empty space
                    
                    # Calculate the center of the rectangle
                    center_x = rectangle.bounds[0] - grid_width 
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
                    
                    # Check if the new line intersects the union
                    if new_line.intersects(region2_union1):
                        # If the new line intersects the union, keep only the part of the line that is within the union
                        new_line_in_union = new_line.intersection(region2_union1)
            
                        # Check if the new line intersects any existing lines
                        intersects_existing_line = any(new_line_in_union.intersects(line) for line in lines_in_region.geometry)
                        if not intersects_existing_line:
                            
                            # Check if the new line intersects any new lines
                            new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)
                            intersects_new_line = any(new_line_in_union.intersects(line) for line in new_lines_gdf.geometry)
                            if not intersects_new_line:
                                # If the new line does not intersect any existing lines, add it to the list of new lines
                                new_lines.append(new_line_in_union)
                    
                    # Iterate over each new line
                    for line in new_lines:
                        # Check if the rectangle intersects with the line and if the intersection is a LineString
                        if rectangle.intersects(line) and isinstance(rectangle.intersection(line), LineString):
                            # Calculate the length of the intersection
                            intersection_length = rectangle.intersection(line).length
            
                            # Add the length of the intersection to the total
                            total_intersection_length += intersection_length
                            
                            # Increment the counter
                            lines_in_this_square += 1
            
                    # Check if the total length of intersections is less than half of the rectangle's perimeter
                    if total_intersection_length < intersection_area_length / 5: #and lines_in_this_square < 100:
                        ########################## The rectangle is in an empty space
                        
                        # Calculate the lower left corner of the rectangle
                        lower_left_x = rectangle.bounds[0]
                        lower_left_y = rectangle.bounds[1]
            
                        # Calculate the upper left corner of the rectangle
                        # Calculate the end point of the line based on the rotation angle
                        upper_left_x = lower_left_x + median_length * np.cos(rotation_angle)
                        upper_left_y = lower_left_y + median_length * np.sin(rotation_angle)
            
                        # Calculate the end points of the line based on the rotation angle
                        start_x = lower_left_x
                        start_y = lower_left_y
                        end_x = upper_left_x
                        end_y = upper_left_y
            
                        #Create the new line
                        new_line = LineString([(start_x, start_y), (end_x, end_y)])
                        
                        # Check if the new line intersects the union
                        if new_line.intersects(region2_union1):
                            # If the new line intersects the union, keep only the part of the line that is within the union
                            new_line_in_union = new_line.intersection(region2_union1)
            
                            # Check if the new line intersects any existing lines
                            intersects_existing_line = any(new_line_in_union.intersects(line) for line in lines_in_region.geometry)
                            if not intersects_existing_line:
                            
                            
                                # Check if the new line intersects any new lines
                                new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)
                                intersects_new_line = any(new_line_in_union.intersects(line) for line in new_lines_gdf.geometry)
                                if not intersects_new_line:
                                    # If the new line does not intersect any existing lines, add it to the list of new lines
                                    new_lines.append(new_line_in_union)
            
                        # Iterate over each new line
                        for line in new_lines:
                            # Check if the rectangle intersects with the line and if the intersection is a LineString
                            if rectangle.intersects(line) and isinstance(rectangle.intersection(line), LineString):
                                # Calculate the length of the intersection
                                intersection_length = rectangle.intersection(line).length
            
                                # Add the length of the intersection to the total
                                total_intersection_length += intersection_length
                                
                                # Increment the counter
                                lines_in_this_square += 1
            
                        # Check if the total length of intersections is less than half of the rectangle's perimeter
                        if total_intersection_length < intersection_area_length / 5: #lines_in_this_square < 100:
                            ################## The rectangle is in an empty space
                            
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
                    
                            # Check if the new line intersects the union
                            if new_line.intersects(region2_union1):
                                # If the new line intersects the union, keep only the part of the line that is within the union
                                new_line_in_union = new_line.intersection(region2_union1)
            
                                # Check if the new line intersects any existing lines
                                intersects_existing_line = any(new_line_in_union.intersects(line) for line in lines_in_region.geometry)
                                if not intersects_existing_line:
                            
                                    # Check if the new line intersects any new lines
                                    new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)
                                    intersects_new_line = any(new_line_in_union.intersects(line) for line in new_lines_gdf.geometry)
                                    if not intersects_new_line:
                                        # If the new line does not intersect any existing lines, add it to the list of new lines
                                        new_lines.append(new_line_in_union)
            
                
            
                    
                                                        #end clustering 
        
            # Convert the list of new lines into a GeoDataFrame
            new_lines_gdf = gpd.GeoDataFrame(geometry=new_lines)
            
            # Plot the old lines and the new lines
            fig, ax = plt.subplots(figsize=(10, 10))
            area_gdf.plot(ax=ax, color='blue')
            #grid.plot(ax=ax, color='black')
            lines_in_region.plot(ax=ax, color='red')
            new_lines_gdf.plot(ax=ax, color='yellow')
            
            
            # Plot the grid points
            #grid.plot(ax=ax, color='black', markersize=10)
            #region_coords = [(-102.9, 31), (-103.4, 31), (-103.4, 31.5), (-102.9, 31.5)]
            
            # After generating the plot, adjust the x and y limits to match the bounds of the region_polygon
            plt.xlim([region1.bounds[0], region1.bounds[2]])
            plt.ylim([region1.bounds[1], region1.bounds[3]])
            
            #plt.xlim([-102.9, -103.4])
            #plt.ylim([31,31.3])
            # Save the plot with a unique name based on the index
            plt.savefig(f'U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/dbk_full__{index}')
            plt.show()
            
        
        
            ## final calculations 
            lengths2 = new_lines_gdf.geometry.length
            t = len(new_lines)
            y = np.average(lengths2)
            total_new_well_length = y*t
        
            # number of net new lines 
            num_new_lines = total_new_well_length / median_length
        
            number_of_new_lines.append(num_new_lines)
        
        else:
            # Plot the existing lines and area
            fig, ax = plt.subplots()
            area_gdf.plot(ax=ax, color='blue')
            plt.xlim([region1.bounds[0], region1.bounds[2]])
            plt.ylim([region1.bounds[1], region1.bounds[3]])
            plt.savefig(f'U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/dbk_example_big_{index}')
            plt.show()
            
# Convert the list to a pandas Series
number_of_new_lines = pd.Series(number_of_new_lines)

number_of_new_lines = number_of_new_lines.dropna()

print(number_of_new_lines)
print(sum(number_of_new_lines))

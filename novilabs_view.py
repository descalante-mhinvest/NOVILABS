# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:46:52 2023

@author: descalante
"""

#NOVILABS

import pandas as pd
import geopandas as gpd
from shapely import wkt

df = pd.read_csv('S:/IT/NoviLabs/Novi_v2_US_Horizontals_T1_All_2023-03-01/Database/CompanyAcreage.tsv', sep='\t')
df['geometry'] = df.Geometry.apply(wkt.loads)
df.drop('Geometry', axis=1, inplace=True) #Drop Geometry column
gdf = gpd.GeoDataFrame(df, geometry='geometry')                     
gdf = gdf[gdf['Company'] == "Magnolia Oil & Gas"]
gdf2 = gdf['geometry']

gdf2.to_file('U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/mgy_areas')


df2 = pd.read_csv('S:/IT/NoviLabs/Novi_v2_US_Horizontals_T1_All_2023-03-01/Database/CompanyAcreage.tsv', sep='\t')
df['geometry'] = df.Geometry.apply(wkt.loads)
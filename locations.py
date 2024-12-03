"""
Load GeoJSON file with a number of named locations defined by polygons. Accept 
a point and return list of locations it is in. 
"""

import json
import numpy as np

import gnss_tools
import shapely


class LocationTagger:
    def __init__(self, geojson_file):
        self.geojson_file = geojson_file
        self.locations_wgs84 = {}
        self.locations_polygons = {}
        self.enu_origin = None

        self.load_geojson(geojson_file)

    def load_geojson(self, geojson_file):
        """
        Load the GeoJSON file containing location polygons in WGS-84
        coordinates and convert them to shapely polygons in ENU coordinates.
        Save also the ENU frame origin.

        Parameters:
        geojson_file (str): The path to the GeoJSON file.
        """

        # Get list of polygons from the GeoJSON file
        with open(geojson_file, "rt") as f:
            data = json.load(f)

        polygon_centers = np.empty((2, 0))

        for feature in data["features"]:
            # Skip non-polygon features
            if feature["geometry"]["type"] != "Polygon":
                continue

            name = feature["properties"]["name"]

            # Swap the coordinates so latitude goes first
            polygon_wgs84 = np.flip(
                np.array(feature["geometry"]["coordinates"])[0].T, axis=0
            )
            self.locations_wgs84[name] = polygon_wgs84
            polygon_centers = np.append(
                polygon_centers, np.mean(polygon_wgs84, axis=1).reshape(2, 1), axis=1
            )

        # Use the mean of the polygon centers as the ENU frame origin
        locations_center = np.mean(polygon_centers, axis=1).reshape(2, 1)
        self.enu_origin = locations_center

        # Convert the polygon coordinates to ENU
        locations = {}
        for name, polygon_wgs84 in self.locations_wgs84.items():
            polygon_enu = gnss_tools.WGS84_to_ENU(
                gnss_tools.prep_coords_wgs84(polygon_wgs84),
                gnss_tools.prep_coords_wgs84(locations_center),
            )
            polygon_shapely = shapely.Polygon(polygon_enu.T)

            if not polygon_shapely.is_valid:
                print("WARN: Invalid polygon \"{}\" in the given GeoJSON file".format(name))
            
            shapely.prepare(polygon_shapely) # speeds up the contains method
            locations[name] = polygon_shapely

        self.locations_polygons = locations

    def tag_points(self, points):
        """
        Assign a list of locations to each given point.

        Parameters:
        points (np.ndarray): A numpy array containing the WGS-84 coordinates
        with (3, N) or (2, N) shape, where the first row is latitude,
        the second row is longitude, and the third row is altitude (if
        present). Latitude and longitude are in degrees, and altitude is in
        meters.
        """

        # Convert the point coordinates to ENU
        points_enu = gnss_tools.WGS84_to_ENU(
            gnss_tools.prep_coords_wgs84(points),
            gnss_tools.prep_coords_wgs84(self.enu_origin),
        )

        # Tag the points
        all_tags = []
        debug_pnt_list = [] ###
        for point_enu in points_enu.T:
            pnt_tags = []
            pnt = shapely.geometry.Point(point_enu)
            debug_pnt_list.append(pnt) ###
            for name, polygon in self.locations_polygons.items():
                if shapely.contains(polygon, pnt):
                    pnt_tags.append(name)

            all_tags.append(pnt_tags)

        return all_tags

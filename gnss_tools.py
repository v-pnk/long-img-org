import numpy as np
import math


# GRS80 ellipsoid:
# - Equatorial Earth radius [m] (semi-major axis)
R_EQU = 6378137.0
# - Polar Earth radius [m] (semi-minor axis)
R_POL = 6356752.314140347
# - squared eccentricity of the ellipsoid
# ECC2 = 1 - (R_POL/R_EQU)**2
ECC2 = 0.00669437999014133
# - ellipsoid flattening
# FLA = 1 - R_POL/R_EQU
FLA = 0.003352810664747463


def prep_coords_wgs84(coords_wgs84):
    """
    Preprocess the coordinates before other operations. Convert latitude and
    longitude to radians. If the altitude is not provided, set it to zero.

    Parameters:
    coords_wgs84 (np.ndarray): A numpy array containing the WGS-84 coordinates
    with (2, N) or (3, N) shape, where the first row is latitude, the second
    row is longitude, and the third row (if present) is altitude. Latitude and
    longitude are in degrees, and altitude is in meters.

    Returns:
    coords_wgs84_prep (np.ndarray): A numpy array containing the preprocessed
    WGS-84 coordinates with shape (3, N).
    """

    # convert latitude and longitude to radians
    coords_wgs84_prep = np.empty(coords_wgs84.shape)
    coords_wgs84_prep[0, :] = np.radians(coords_wgs84[0, :])
    coords_wgs84_prep[1, :] = np.radians(coords_wgs84[1, :])

    if coords_wgs84_prep.shape[0] == 2:
        coords_wgs84_prep = np.vstack(
            (coords_wgs84_prep, np.zeros(coords_wgs84_prep.shape[1]))
        )

    return coords_wgs84_prep


def WGS84_to_ECEF(coords_wgs84):
    """
    Convert WGS-84 coordinates to ECEF (Earth-Centered, Earth-Fixed)
    coordinates.

    Parameters:
    coords_wgs84 (np.ndarray): A numpy array containing the WGS-84 coordinates
    with (3, N) shape, where the first row is latitude, the second row is
    longitude, and the third row is altitude. Latitude and longitude are in
    radians, and altitude is in meters.

    Returns:
    coords_ecef (np.ndarray): A numpy array containing the ECEF coordinates
    with shape (3, N).
    """

    # prime vertical radius of curvature
    N = R_EQU / np.sqrt(1 - ECC2 * np.sin(coords_wgs84[0, :]) ** 2)

    # compute the ECEF coordinates
    x = (
        (N + coords_wgs84[2, :])
        * np.cos(coords_wgs84[0, :])
        * np.cos(coords_wgs84[1, :])
    )
    y = (
        (N + coords_wgs84[2, :])
        * np.cos(coords_wgs84[0, :])
        * np.sin(coords_wgs84[1, :])
    )
    z = ((1 - ECC2) * N + coords_wgs84[2, :]) * np.sin(coords_wgs84[0, :])

    return np.array([x, y, z])


def ECEF_to_ENU(coords_ecef, origin_coords_wgs84):
    """
    Convert ECEF coordinates to a local tangent plane.
    Use ENU coordinate frame orientation (East, North, Up).

    Parameters:
    coords_ecef (np.ndarray): A numpy array containing the ECEF coordinates
    with (3, N) shape.
    origin_coords_wgs84 (np.ndarray): A numpy array containing the WGS-84
    coordinates of the origin with (3,) shape. The first element is latitude,
    the second element is longitude, and the third element is altitude.
    Latitude and longitude are in radians, and altitude is in meters.

    Returns:
    coords_enu: A numpy array containing the ENU coordinates with shape (3, N).
    """

    sin_lat_orig = math.sin(origin_coords_wgs84[0, 0])
    cos_lat_orig = math.cos(origin_coords_wgs84[0, 0])
    sin_lon_orig = math.sin(origin_coords_wgs84[1, 0])
    cos_lon_orig = math.cos(origin_coords_wgs84[1, 0])

    # rotation from ECEF to ENU
    R = np.array(
        [
            [-sin_lon_orig, cos_lon_orig, 0],
            [-cos_lon_orig * sin_lat_orig, -sin_lon_orig * sin_lat_orig, cos_lat_orig],
            [cos_lon_orig * cos_lat_orig, sin_lon_orig * sin_lat_orig, sin_lat_orig],
        ]
    )

    # compute the differences
    origin_coords_ecef = WGS84_to_ECEF(origin_coords_wgs84)
    diff_ecef = coords_ecef - origin_coords_ecef

    # compute the ENU coordinates
    coords_enu = np.dot(R, diff_ecef)

    return coords_enu


def WGS84_to_ENU(coords_wgs84, origin_coords_wgs84):
    """
    Convert WGS-84 coordinates to a local tangent plane.
    Use ENU coordinate frame orientation (East, North, Up).

    Parameters:
    coords_wgs84 (np.ndarray): A numpy array containing the WGS-84 coordinates
    with (3, N) shape, where the first row is latitude, the second row is
    longitude, and the third row is altitude. Latitude and longitude are in
    radians, and altitude is in meters.

    Returns:
    coords_enu (np.ndarray): A numpy array containing the local tangent plane
    ENU-oriented coordinates with shape (3, N).
    """

    coords_ecef = WGS84_to_ECEF(coords_wgs84)
    coords_enu = ECEF_to_ENU(coords_ecef, origin_coords_wgs84)

    return coords_enu


def get_R_GEO(lat):
    """
    Compute the Earth geocentric radius at the given latitude.

    Parameters:
    lat (np.ndarray): Array of latitudes in radians.

    Returns:
    r_geo (np.ndarray): Array of the Earth geocentric radii.
    """

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    return np.sqrt(
        (R_EQU**4 * cos_lat**2 + R_POL**4 * sin_lat**2)
        / (R_EQU**2 * cos_lat**2 + R_POL**2 * sin_lat**2)
    )


def hav(ang):
    """
    Haversine function.

    Parameters:
    ang (np.ndarray): Array of angles in radians.

    Returns:
    h (np.ndarray): Array of Haversine function values.
    """

    s = np.sin(0.5 * ang)
    return s * s


def ahav(h):
    """
    Inverse haversine function.

    Parameters:
    h (np.ndarray): Array of Haversine function values.

    Returns:
    ang (np.ndarray): Array of angles in radians.
    """

    return 2 * np.arcsin(np.sqrt(h))


def dist_haversine(p1, p2, alt=False):
    """
    Metrical distance of two points on Earth surface using haversine formula.

    Parameters:
    p1 (np.ndarray): First point coordinates in the form
    [[latitude], [longitude]] or [[latitude], [longitude], [altitude]].
    Latitude and longitude are in radians, and altitude is in meters.
    p2 (np.ndarray): Second point coordinates in the form
    [[latitude], [longitude]] or [[latitude], [longitude], [altitude]].
    Latitude and longitude are in radians, and altitude is in meters.
    alt (bool): If True, the altitude is included in the distance calculation.

    Returns:
    dist (np.ndarray): Metric distance between p1 and p2.
    """

    lat1 = p1[0, :]
    lon1 = p1[1, :]
    lat2 = p2[0, :]
    lon2 = p2[1, :]

    lat_avg = 0.5 * (lat1 + lat2)
    R_GEO = get_R_GEO(lat_avg)

    if alt:
        alt1 = p1[2, :]
        alt2 = p2[2, :]
        alt_avg = 0.5 * (alt1 + alt2)
        R_GEO = R_GEO + alt_avg

        d = R_GEO * ahav(
            hav(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * hav(lon2 - lon1)
        )
        return np.sqrt(d * d + (alt2 - alt1) * (alt2 - alt1))

    else:
        return R_GEO * ahav(
            hav(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * hav(lon2 - lon1)
        )

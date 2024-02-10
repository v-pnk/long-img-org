"""
EXIF helper functions
"""


import datetime
import numpy as np


def exif_to_sensor_info(exif_tags: dict):
    """Get sensor information from the EXIF data. Compose also the sensor name
    in the following format:
    <camera_model>_<focal_length>mm_<resolution_x>x<resolution_y>_<data_type>

    The data type is RGB by default. TODO: add others like DEPTH or RAW

    Parameters:
    exif_tags (dict): The loaded EXIF tags from an image.

    Returns:
    sensor_name (str): The sensor name.
    sensor_info (dict): The sensor information.

    """

    device = exif_tags["EXIF:Model"]
    focal_length = exif_tags["EXIF:FocalLength"]
    resolution_x = exif_tags["EXIF:ImageWidth"]
    resolution_y = exif_tags["EXIF:ImageHeight"]
    data_type = "RGB"

    sensor_name = "{}_{}mm_{}x{}_{}".format(
        device, focal_length, resolution_x, resolution_y, data_type
    )

    sensor_info = {
        "device": device,
        "focal_length": focal_length,
        "resolution_x": resolution_x,
        "resolution_y": resolution_y,
        "data_type": data_type,
    }

    return sensor_name, sensor_info


def exif_to_time(exif_tags: dict):
    """Get the image capture time from EXIF tags.

    Parameters:
    exif_tags (dict): The loaded EXIF tags from an image.

    Returns:
    full_datetime (datetime): The full date and time of the image capture.

    """

    capture_datetime = exif_tags["EXIF:DateTimeOriginal"]
    if "EXIF:SubSecTimeOriginal" in exif_tags:
        capture_subsec = str(exif_tags["EXIF:SubSecTimeOriginal"])
        capture_millisecond = capture_subsec[:3]
    else:
        capture_millisecond = "000"
    capture_tz_offset = exif_tags["EXIF:OffsetTimeOriginal"]

    tz = datetime.datetime.strptime(capture_tz_offset, "%z")

    full_datetime = datetime.datetime.strptime(capture_datetime, "%Y:%m:%d %H:%M:%S")
    full_datetime = full_datetime.replace(microsecond=1000 * int(capture_millisecond))
    full_datetime = full_datetime.replace(tzinfo=tz.tzinfo)

    return full_datetime


def time_to_exif(full_datetime: datetime.datetime):
    """Convert the full date and time to EXIF tags.

    Parameters:
    full_datetime (datetime): The full date and time of the image capture.

    Returns:
    exif_tags (dict): The EXIF date and time tags.

    """

    capture_datetime = full_datetime.strftime("%Y:%m:%d %H:%M:%S")
    capture_millisecond = full_datetime.strftime("%f")[:3]
    capture_tz_offset = full_datetime.strftime("%z")
    capture_tz_offset = capture_tz_offset[:3] + ":" + capture_tz_offset[3:]

    exif_tags = {
        "EXIF:DateTimeOriginal": capture_datetime,
        "EXIF:SubSecTimeOriginal": capture_millisecond,
        "EXIF:OffsetTimeOriginal": capture_tz_offset,
    }

    return exif_tags


def exif_to_WGS84(exif_tags: dict):
    """Get the GNSS coordinates from the EXIF tags.

    Parameters:
    exif_tags (dict): The loaded EXIF tags from an image.

    Returns:
    coords_wgs84 (np.ndarray): The GNSS coordinates.

    """

    if "EXIF:GPSLatitude" not in exif_tags:
        return None

    lat = exif_tags["EXIF:GPSLatitude"]
    lon = exif_tags["EXIF:GPSLongitude"]

    lat_ref = exif_tags["EXIF:GPSLatitudeRef"]
    lon_ref = exif_tags["EXIF:GPSLongitudeRef"]

    if lat_ref == "S":
        lat = -lat

    if lon_ref == "W":
        lon = -lon

    if "EXIF:GPSAltitude" in exif_tags and "EXIF:GPSAltitudeRef" in exif_tags:
        alt = exif_tags["EXIF:GPSAltitude"]
        alt_ref = exif_tags["EXIF:GPSAltitudeRef"]

        if alt_ref == 1:
            alt = -alt
    else:
        alt = 0.0

    return np.array([[lat], [lon], [alt]])

def WGS84_to_exif(coords: np.ndarray):
    """Convert GNSS coordinates to EXIF tags.

    Parameters:
    coords (np.ndarray): The GNSS coordinates with size (2,) or (3,).

    Returns:
    exif_tags (dict): The EXIF GPS-related tags.

    """

    coords = coords.flatten()

    lat = coords[0]
    lon = coords[1]

    lat_ref = "N"
    lon_ref = "E"

    if lat < 0:
        lat_ref = "S"
        lat = -lat

    if lon < 0:
        lon_ref = "W"
        lon = -lon

    exif_tags = {
        "EXIF:GPSLatitude": lat,
        "EXIF:GPSLatitudeRef": lat_ref,
        "EXIF:GPSLongitude": lon,
        "EXIF:GPSLongitudeRef": lon_ref,
    }

    if coords.shape[0] == 2:
        alt = coords[2]
        alt_ref = 0

        if alt < 0:
            alt_ref = 1
            alt = -alt
        
        exif_tags["EXIF:GPSAltitude"] = alt
        exif_tags["EXIF:GPSAltitudeRef"] = alt_ref

    return exif_tags
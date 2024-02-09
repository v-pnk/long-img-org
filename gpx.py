"""
Helper functions for GPX file loading and interpolation.
"""


import xml.etree.ElementTree as ET
import datetime
import numpy as np


def load_gpx_file(gpx_file: str):
    """Parse the GPX file and get the time series of GNSS coordinates.

    Parameters:
    gpx_file (str): The path to the GPX file.

    Returns:
    timestamps (np.ndarray): The time series of the GNSS coordinates in UTC.
    coords_wgs84 (np.ndarray): The GNSS coordinates.

    """

    timestamps = np.empty((0,), dtype=datetime.datetime)
    coords_wgs84 = np.empty((3, 0))

    tree = ET.parse(gpx_file)
    root = tree.getroot()
    schema = root.attrib[
        "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"
    ].split()[0]

    for trkpt in root.findall(".//{" + schema + "}trkpt"):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])

        alt = float(trkpt.find("{" + schema + "}ele").text)

        # The time in GPX should be in UTC / GMT time zone
        time_str = trkpt.find("{" + schema + "}time").text
        time_str_crop = time_str[:19]  # crop the subsecond part and Z
        time = datetime.datetime.strptime(time_str_crop, "%Y-%m-%dT%H:%M:%S")
        if len(time_str) > 20:
            # add the microseconds
            time = time.replace(microsecond=1000 * int(time_str[20:23]))
            print("adding_microseconds: " + time_str[20:23])

        timestamps = np.append(timestamps, time)
        coords_wgs84 = np.append(coords_wgs84, np.array([[lat], [lon], [alt]]), axis=1)

    return timestamps, coords_wgs84


def gpx_interpolate(timestamps, coords_wgs84, capture_time, mode="nearest"):
    """Interpolate the GNSS coordinates for the given capture time.

    Parameters:
    timestamps (np.ndarray): The time series of the GNSS coordinates in UTC.
    coords_wgs84 (np.ndarray): The GNSS coordinates.
    capture_time (datetime): The capture time.
    mode (str): The mode of the interpolation. The options are "nearest",
        "nearest_outside", "linear", and "linear_outside".


    Returns:
    coords_wgs84 (np.ndarray): The interpolated GNSS coordinates.

    """

    min_time = timestamps[0]
    max_time = timestamps[-1]

    capture_time_utc = capture_time.astimezone(datetime.timezone.utc).replace(
        tzinfo=None
    )

    if capture_time_utc < min_time or capture_time_utc > max_time:
        if mode in ["nearest_outside", "linear_outside"]:
            if capture_time_utc < min_time:
                return coords_wgs84[0]
            else:
                return coords_wgs84[-1]
        else:
            raise ValueError("The capture time is outside the GPX time range.")

    if mode in ["nearest", "nearest_outside"]:
        nearest_index = np.argmin(np.abs(timestamps - capture_time_utc))
        return coords_wgs84[nearest_index]
    elif mode in ["linear", "linear_outside"]:
        if timestamps[nearest_index] < capture_time_utc:
            t1 = timestamps[nearest_index]
            t2 = timestamps[nearest_index + 1]
            c1 = coords_wgs84[nearest_index]
            c2 = coords_wgs84[nearest_index + 1]
        else:
            t1 = timestamps[nearest_index - 1]
            t2 = timestamps[nearest_index]
            c1 = coords_wgs84[nearest_index - 1]
            c2 = coords_wgs84[nearest_index]

        dt = (capture_time_utc - t1) / (t2 - t1)
        return c1 + dt * (c2 - c1)

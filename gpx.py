"""
Helper functions for GPX file loading and interpolation.
"""


import xml.etree.ElementTree as ET
import datetime
import numpy as np


def load_gpx_file(gpx_file: str, ):
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

        # Time can have multiple formats
        # - standard format in UTC time zone:   2008-07-18T14:07:50.000Z
        # - format with time zone:              2008-07-18T16:07:50.000+02:00
        time_str = trkpt.find("{" + schema + "}time").text
        time_str_crop = time_str[:19]  # crop only the date and time part
        time = datetime.datetime.strptime(time_str_crop, "%Y-%m-%dT%H:%M:%S")
        
        # If the time string has subsecond and/or timezone information
        if len(time_str) > 20:
            time_str_crop = time_str[19:]
            dot_idx = time_str_crop.find(".")
            sign_idx = time_str_crop.find("+") if "+" in time_str_crop else time_str_crop.find("-")

            if sign_idx != -1:
                tz_str = time_str_crop[sign_idx:].replace(":", "")
                if dot_idx != -1:
                    subsec_str = time_str_crop[dot_idx + 1:sign_idx].replace("Z", "")
                else:
                    subsec_str = "0"
            else:
                tz_str = "+0000"
                if dot_idx != -1:
                    subsec_str = time_str_crop[dot_idx + 1:].replace("Z", "")
                else:
                    subsec_str = "0"
            
            time = time.replace(microsecond=1000 * int(subsec_str))
            tz = datetime.datetime.strptime(tz_str, "%z")
            tz = datetime.timezone(tz.utcoffset())
            time = time.replace(tzinfo=tz)

            # Convert to UTC
            time = time.astimezone(datetime.timezone.utc).replace(tzinfo=None)

        timestamps = np.append(timestamps, time)
        coords_wgs84 = np.append(coords_wgs84, np.array([[lat], [lon], [alt]]), axis=1)

    return timestamps, coords_wgs84


def gpx_interpolate_mult(timestamps, coords_wgs84, capture_time, mode="nearest"):
    """Interpolate the GNSS coordinates for the given capture time and multiple
    GPX time series.

    Parameters:
    timestamps (Union[np.ndarray, list[np.ndarray]]): The time series of 
        the GNSS coordinates in UTC. If multiple time series are given,
        we assume that the time series are not overlapping.
    coords_wgs84 (Union[np.ndarray, list[np.ndarray]]): The GNSS coordinates.
    capture_time (datetime): The capture time.
    mode (str): The mode of the interpolation. The options are "nearest",
        "nearest_outside", "linear", and "linear_outside".


    Returns:
    coords_wgs84 (np.ndarray): The interpolated GNSS coordinates.

    """

    if type(timestamps) is not list:
        assert type(coords_wgs84) is not list
        timestamps = [timestamps]
        coords_wgs84 = [coords_wgs84]

    assert len(timestamps) == len(coords_wgs84)

    min_times = np.array([ts[0] for ts in timestamps])
    max_times = np.array([ts[-1] for ts in timestamps])

    # Convert to UTC time zone
    # - if .astimezone() would be called on a naive datetime object (without 
    #   time zone), it would assume the system's local timezone
    if capture_time.tzinfo is not None:
        capture_time_utc = capture_time.astimezone(datetime.timezone.utc).replace(
            tzinfo=None
        )
    else:
        capture_time_utc = capture_time
    
    nearest_min_ts_idx = np.argmin(np.abs(min_times - capture_time_utc))
    nearest_max_ts_idx = np.argmin(np.abs(max_times - capture_time_utc))
    nearest_min_dt = np.min(np.abs(min_times - capture_time_utc))
    nearest_max_dt = np.min(np.abs(max_times - capture_time_utc))

    if nearest_min_dt < nearest_max_dt:
        return gpx_interpolate(
            timestamps[nearest_min_ts_idx],
            coords_wgs84[nearest_min_ts_idx],
            capture_time_utc,
            mode,
        )
    else:
        return gpx_interpolate(
            timestamps[nearest_max_ts_idx],
            coords_wgs84[nearest_max_ts_idx],
            capture_time_utc,
            mode,
        )
        

def gpx_interpolate(timestamps, coords_wgs84, capture_time, mode="nearest"):
    """Interpolate the GNSS coordinates for the given capture time and single
    GPX time series.

    Parameters:
    timestamps (np.ndarray): The time series of the GNSS coordinates in UTC.
    coords_wgs84 (np.ndarray): The GNSS coordinates.
    capture_time (datetime): The capture time.
    mode (str): The mode of the interpolation. The options are "nearest",
        "nearest_outside", "linear", and "linear_outside".

    Returns:
    coords_wgs84 (np.ndarray): The interpolated GNSS coordinates with (3x1) shape.

    """

    min_time = timestamps[0]
    max_time = timestamps[-1]

    # Convert to UTC time zone
    # - if .astimezone() would be called on a naive datetime object (without 
    #   time zone), it would assume the system's local timezone
    if capture_time.tzinfo is not None:
        capture_time_utc = capture_time.astimezone(datetime.timezone.utc).replace(
            tzinfo=None
        )
    else:
        capture_time_utc = capture_time

    if capture_time_utc < min_time or capture_time_utc > max_time:
        if mode in ["nearest_outside", "linear_outside"]:
            if capture_time_utc < min_time:
                return np.reshape(coords_wgs84[:, 0], (3, 1))
            else:
                return np.reshape(coords_wgs84[:, -1], (3, 1))
        else:
            assert False, "The capture time is outside the GPX time range: time = {}, range = ({} - {})".format(capture_time_utc, min_time, max_time)

    nearest_index = np.argmin(np.abs(timestamps - capture_time_utc))

    if mode in ["nearest", "nearest_outside"]:
        return np.reshape(coords_wgs84[:, nearest_index], (3,1))
    elif mode in ["linear", "linear_outside"]:
        if timestamps[nearest_index] < capture_time_utc:
            t1 = timestamps[nearest_index]
            t2 = timestamps[nearest_index + 1]
            c1 = coords_wgs84[:, nearest_index]
            c2 = coords_wgs84[:, nearest_index + 1]
        else:
            t1 = timestamps[nearest_index - 1]
            t2 = timestamps[nearest_index]
            c1 = coords_wgs84[:, nearest_index - 1]
            c2 = coords_wgs84[:, nearest_index]

        dt = (capture_time_utc - t1) / (t2 - t1)
        return np.reshape(c1 + dt * (c2 - c1), (3, 1))

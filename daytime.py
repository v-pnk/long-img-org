"""
Compute the daytime for a given location and time.
Based on https://gml.noaa.gov/grad/solcalc/solareqns.PDF.
"""


import datetime
import math


def get_sunset_sunrise(lat, lon, date, type="official"):
    """
    Compute the sunset and sunrise times for the given location and date.

    Parameters:
    lat (float): Latitude in degrees.
    lon (float): Longitude in degrees.
    date (datetime.datetime): Date and time.
    type (str): Type of the sunrise and sunset times. Can be "official",
    "civil", "nautical", or "astronomical". Default is "official".


    Returns:
    sunrise (datetime.datetime): Sunrise time.
    sunset (datetime.datetime): Sunset time.
    """

    # Fractional year
    day_of_year = date.timetuple().tm_yday

    if date.year % 4 == 0:
        days_in_year = 366
    else:
        days_in_year = 365

    gamma = 2 * math.pi / days_in_year * (day_of_year - 1 + (date.hour - 12) / 24)

    # Equation of time (in minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # Solar declination angle (in radians)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    # Time offset (in minutes)
    timezone_minutes = date.utcoffset().total_seconds() / 60
    time_offset = eqtime + 4 * lon - timezone_minutes

    # True solar time (in minutes)
    tst = date.hour * 60 + date.minute + date.second / 60 + time_offset

    # Zenith based on the type of the sunrise and sunset times
    # - use an approximation without refraction and solar disk size corrections
    if type == "official":  # sun at horizon
        zenith = math.radians(90)
    elif type == "civil":  # sun 6 degrees below horizon
        zenith = math.radians(96)
    elif type == "nautical":  # sun 12 degrees below horizon
        zenith = math.radians(102)
    elif type == "astronomical":  # sun 18 degrees below horizon
        zenith = math.radians(108)
    else:
        raise ValueError("Invalid type of the sunrise and sunset times.")

    # Solar hour angle (in radians)
    ha = math.acos(
        (math.cos(zenith) / (math.cos(math.radians(lat)) * math.cos(decl)))
        - math.tan(math.radians(lat)) * math.tan(decl)
    )

    # Sunrise and sunset times (in minutes)
    sunrise = 720 - 4 * (lon + math.degrees(ha)) - eqtime + timezone_minutes
    sunset = 720 - 4 * (lon - math.degrees(ha)) - eqtime + timezone_minutes

    # Convert the times to datetime objects
    sunrise = datetime.datetime.combine(
        date.date(),
        datetime.time(int(sunrise // 60), int(sunrise % 60)),
        tzinfo=date.tzinfo,
    )
    sunset = datetime.datetime.combine(
        date.date(),
        datetime.time(int(sunset // 60), int(sunset % 60)),
        tzinfo=date.tzinfo,
    )

    return sunrise, sunset


def get_daytime_tag(lat, lon, dt):
    """
    Compute the daytime tag for the given location and time.

    Parameters:
    lat (float): Latitude in degrees.
    lon (float): Longitude in degrees.
    dt (datetime.datetime): Date and time.

    Returns:
    tag (str): Daytime tag for the given day and time. Can be "day", "night",
    "dawn", or "dusk".
    """

    # Get the sunrise and sunset times
    sunrise_official, sunset_official = get_sunset_sunrise(
        lat, lon, dt, type="official"
    )
    sunrise_civil, sunset_civil = get_sunset_sunrise(lat, lon, dt, type="civil")

    # Define the dawn and dusk as the time between the civil and official
    # sunrise or sunset
    if sunrise_civil < dt < sunrise_official:
        tag = "dawn"
    elif sunset_official < dt < sunset_civil:
        tag = "dusk"
    elif sunrise_official < dt < sunset_official:
        tag = "day"
    else:
        tag = "night"

    return tag

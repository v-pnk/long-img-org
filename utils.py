"""
Various utility functions used in the project, not falling into any specific
category.
"""


import os
import datetime
import csv

import yaml
import numpy as np
import cv2
import exiftool


def time_to_name(full_datetime: datetime):
    """Get the image name from the creation date and time. The image name is in
    the following format:
    <year>-<month>-<day>_<hour>-<minute>-<second>-<millisecond>_<timezone>

    The timezone is the UTC offset in <sign><hours><minutes> format, where
    the sign can be P for plus or M for minus.

    Parameters:
    full_datetime (datetime): The full date and time of the image capture.

    Returns:
    image_name (str): The image name.

    """

    # Prepare the timezone string
    datetime_str = full_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    millis_str = "{:03d}".format(int(0.001 * full_datetime.microsecond))

    tz_str = full_datetime.strftime("%z")
    if tz_str[0] == "+":
        tz_str = "P" + tz_str[1:]
    else:
        tz_str = "M" + tz_str[1:]
    tz_str = tz_str[:3] + "-" + tz_str[3:]

    image_name = "{}-{}_{}".format(datetime_str, millis_str, tz_str)

    return image_name


def tag_sequence(image_dict, min_fraction=0.2):
    """
    Tag the sequence based on the image tags.

    Parameters:
    image_dict (dict): A list of images with metadata.
    min_fraction (float): The minimum fraction of the images with the same tag
        to assign the tag to the sequence.

    Returns:
    tag_location (list): The list of location tags.
    tag_daytimes (list): The list of day-time tags.
    TODO: tag_weather (list): The list of weather tags.

    """

    tag_location = []
    tag_daytimes = []
    # tag_weather = []

    for image in image_dict.values():
        tag_location += image["tag_location"]
        tag_daytimes.append(image["tag_daytime"])
        # tag_weather += image["tag_weather"]

    # Count occurences of individual tags
    tag_location_counts = {}
    for tag in tag_location:
        if tag in tag_location_counts:
            tag_location_counts[tag] += 1
        else:
            tag_location_counts[tag] = 1

    tag_daytime_counts = {}
    for tag in tag_daytimes:
        if tag in tag_daytime_counts:
            tag_daytime_counts[tag] += 1
        else:
            tag_daytime_counts[tag] = 1

    # tag_weather_counts = {}
    # for tag in tag_weather:
    #     if tag in tag_weather_counts:
    #         tag_weather_counts[tag] += 1
    #     else:
    #         tag_weather_counts[tag] = 1

    # Keep only tags that meet the fraction criteria
    tag_location = [
        tag
        for tag in tag_location_counts
        if tag_location_counts[tag] >= min_fraction * len(image_dict)
    ]
    tag_daytimes = [
        tag
        for tag in tag_daytime_counts
        if tag_daytime_counts[tag] >= min_fraction * len(image_dict)
    ]
    # tag_weather = [tag for tag in tag_weather_counts
    #     if tag_weather_counts[tag] >= min_fraction * len(image_dict)]

    return tag_location, tag_daytimes


def load_metadata(metadata_file):
    """Load the metadata file.

    Parameters:
    metadata_file (str): The path to the metadata file.

    Returns:
    metadata (dict): The metadata.

    """

    with open(metadata_file, "rt") as f:
        metadata = yaml.safe_load(f)

    for seq in metadata["sequences"].values():
        for image_name in seq["images"]:
            seq["images"][image_name]["coords_wgs84"] = np.array(
                seq["images"][image_name]["coords_wgs84"]
            ).reshape(3, 1)

    return metadata


def save_metadata(metadata_file, metadata):
    """Save the metadata file.

    Parameters:
    metadata_file (str): The path to the metadata file.
    metadata (dict): The metadata.

    """

    metadata_copy = metadata.copy()
    for seq in metadata_copy["sequences"].values():
        for image_name in seq["images"]:
            seq["images"][image_name]["coords_wgs84"] = (
                seq["images"][image_name]["coords_wgs84"].flatten().tolist()
            )

    with open(metadata_file, "wt") as f:
        yaml.dump(metadata, f)


def copy_resize_image(source_path, target_path, ratio):
    """Copy the image from the source to the target, resize it, save it to the
    target path, and transfer the EXIF data.

    Parameters:
    source_path (str): The source image file.
    target_path (str): The target image file.
    ratio (float): The resize ratio for downsampling.

    """

    image = cv2.imread(source_path)

    # Downsample the frame
    if ratio < 1.0:
        image = cv2.resize(
            image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
        )

    cv2.imwrite(target_path, image)

    # Transfer the EXIF data
    with exiftool.ExifToolHelper() as et:
        exif_tags = et.get_metadata(source_path)[0]

        # Copy only valid tag groups
        # - skip "File:" group, which is created automatically
        new_tags_keys = [t for t in exif_tags.keys() if t.startswith(
            ("EXIF:", "MakerNotes:")
            )]
        new_tags = dict((k, exif_tags[k]) for k in new_tags_keys)

        # Fix the image size in the EXIF data
        new_tags["EXIF:ImageWidth"] =  round(
            exif_tags["EXIF:ImageWidth"] * ratio)
        new_tags["EXIF:ImageHeight"] =  round(
            exif_tags["EXIF:ImageHeight"] * ratio)
        
        new_tags["EXIF:ExifImageWidth"] = new_tags["EXIF:ImageWidth"]
        new_tags["EXIF:ExifImageHeight"] = new_tags["EXIF:ImageHeight"]
        
        et.set_tags(target_path, new_tags, params=["-overwrite_original"])


def load_image_database(image_database_file):
    image_database = {}
    with open(image_database_file, 'rt', newline='') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            img_name = row[0]
            date = datetime.datetime.strptime(row[1] + " " + row[2] + " " + row[3], "%Y-%m-%d %H:%M:%S %z")
            sensor_name = row[3]
            sequence = row[4]
            tag_location = row[5]
            tag_daytime = row[6]
            if row[7] == "-":
                coords_wgs84 = None
            else:
                latitude = float(row[7])
                longitude = float(row[8])
                altitude = float(row[9])
                coords_wgs84 = np.array([latitude, longitude, altitude]).reshape(3, 1)
            orig_width = int(row[10])
            orig_height = int(row[11])

            image_database[img_name] = {
                "capture_time": date,
                "sensor_name": sensor_name,
                "sequence": sequence,
                "tag_location": tag_location,
                "tag_daytime": tag_daytime,
                "coords_wgs84": coords_wgs84,
                "orig_width": orig_width,
                "orig_height": orig_height
            }

    return image_database


def save_image_database(image_database_file, image_database):
    if os.path.exists(image_database_file):
        file_mode = "at"
    else:
        file_mode = "wt"

    with open(image_database_file, mode=file_mode, newline='') as f:
        csv_writer = csv.writer(f)

        image_name_list = list(image_database.keys())
        image_name_list.sort()

        for img_name in image_name_list:
            img_data = image_database[img_name]
            date = img_data["capture_time"].strftime("%Y-%m-%d")
            time = img_data["capture_time"].strftime("%H:%M:%S.%f")[:-3]
            timezone = img_data["capture_time"].strftime("%z")
            sensor_name = img_data["sensor_name"]
            sequence = img_data["sequence"]
            tag_location = img_data["tag_location"]
            tag_daytime = img_data["tag_daytime"]
            if "coords_wgs84" not in img_data:
                latitude = "-"
                longitude = "-"
                altitude = "-"
            else:
                latitude = "{:.8f}".format(img_data["coords_wgs84"][0,0])
                longitude = "{:.8f}".format(img_data["coords_wgs84"][1,0])
                altitude = "{:.4f}".format(img_data["coords_wgs84"][2,0])
            orig_width = img_data["orig_width"]
            orig_height = img_data["orig_height"]

            row_data = [img_name, date, time, timezone, sensor_name, sequence, tag_location, tag_daytime, latitude, longitude, altitude, orig_width, orig_height]
            csv_writer.writerow(row_data)


def load_image_database_relpaths(image_database_file):
    image_database_relpaths = []
    with open(image_database_file, 'rt', newline='') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            image_name = row[0]
            date = row[1]
            sensor_name = row[3]

            image_relpath = os.path.join(date, sensor_name, image_name)
            image_database_relpaths.append(image_relpath)
        
        return image_database_relpaths

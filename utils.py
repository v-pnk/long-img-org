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
        all_names = list(seq["images"].keys())
        for image_name in all_names:
            image_relpath = os.path.join(metadata["date"], metadata["sensor"], image_name)
            seq["images"][image_relpath] = seq["images"].pop(image_name)
            if seq["images"][image_relpath]["coords_wgs84"] is not None:
                seq["images"][image_relpath]["coords_wgs84"] = np.array(
                    seq["images"][image_relpath]["coords_wgs84"]
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
        all_relpaths = list(seq["images"].keys())

        for image_relpath in all_relpaths:
            image_name = os.path.basename(image_relpath)
            seq["images"][image_name] = seq["images"].pop(image_relpath)

            if seq["images"][image_name]["coords_wgs84"] is not None:
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

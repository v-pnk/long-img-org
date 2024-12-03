#!/usr/bin/env python3


"""
This script takes a bunch of captured images and videos, processes them and 
adds them into the given dataset directory.

The processing and organization includes:
- Organizing the images by date and sensor
- Extracting frames from videos
- Processing GPX track data
- Renaming the images based on the capture date and time
- Defining the image sequences based on the capture time and GNSS location
- Tagging the images with a location tag
- Tagging the images with a day-time tag
- TODO: Tagging the images with a weather tag
"""


import os
import shutil
import hashlib

import argparse
import numpy as np
import exiftool
import yaml

import gnss_tools
import exif
import locations
import daytime
import gpx
import video
import utils
from img_db import ImageDatabase


parser = argparse.ArgumentParser(description="Process and organize the data.")
parser.add_argument(
    "images", 
    type=str, 
    help="The directory with the unprocessed images.",
)
parser.add_argument(
    "dataset", 
    type=str, 
    help="The dataset directory.",
)
parser.add_argument(
    "--geojson_file", 
    type=str, 
    help="The geojson file with the location tags.",
)
parser.add_argument(
    "--sensor_file",
    type=str,
    help="The YAML file with the sensor list. The default is <dataset>/sensors.yaml.",
)
parser.add_argument(
    "--new_sensor_file",
    type=str,
    help="The YAML file with the sensor list which will be updated based on the new images. The default is <dataset>/sensors.yaml.",
)
parser.add_argument(
    "--image_database_file",
    type=str,
    help="The CSV file with database of all dataset images. The default is <dataset>/image_database.csv.",
)
parser.add_argument(
    "--new_image_database_file",
    type=str,
    help="The CSV file with database of all dataset images which will be updated based on the new images. The default is <dataset>/image_database.csv.",
)
parser.add_argument(
    "--seq_max_td",
    type=float,
    default=300.0,
    help="The maximum time difference between two subsequent images in seconds to consider them as part of the same sequence.",
)
parser.add_argument(
    "--seq_max_dist",
    type=float,
    default=50.0,
    help="The maximum distance for two images in meters to consider them as part of the same sequence.",
)
parser.add_argument(
    "--gpx_file",
    type=str, 
    help="The GPX file / directory with files with the GNSS data.",
)
parser.add_argument(
    "--gpx_mode",
    type=str,
    choices=["nearest", "nearest_outside", "linear", "linear_outside"],
    default="nearest",
    help="GPX interpolation mode. The \"nearest\" and \"linear\" raise an error if the time is outside the GPX time range.",
)
parser.add_argument(
    "--gpx_overwrite",
    action="store_true",
    help="The GNSS data from GPX has precedence over the EXIF data in the images.",
)
parser.add_argument(
    "--video_fps",
    type=float,
    default=2.0,
    help="The frame rate for the video frame extraction.",
)
parser.add_argument(
    "--image_res_ratio",
    type=float,
    default=1.0,
    help="The resolution ratio used for downsampling images - can be in (0.0 - 1.0) range.",
)
parser.add_argument(
    "--video_res_ratio",
    type=float,
    default=1.0,
    help="The resolution ratio used for downsampling video frames - can be in (0.0 - 1.0) range.",
)


img_exts = (".jpg", ".jpeg", ".png")
video_exts = (".mp4")


def main(args):
    # Load the sensor list
    if not args.sensor_file:
        args.sensor_file = os.path.join(args.dataset, "sensors.yaml")

    if os.path.exists(args.sensor_file):
        print("- load given sensor list: {}".format(args.sensor_file))
        with open(args.sensor_file, "rt") as f:
            sensor_list = yaml.safe_load(f)
    else:
        sensor_list = {}

    # Load the image relative paths from the existing image database
    if not args.image_database_file:
        args.image_database_file = os.path.join(args.dataset, "image_database.csv")

    old_image_database = ImageDatabase(root_path = args.dataset)

    if os.path.exists(args.image_database_file):
        print("- load image database: {}".format(args.image_database_file))
        old_image_database.load(args.image_database_file)
        
    image_database_relpaths = old_image_database.get_relpaths()

    # Get the list of image and video paths
    image_paths = []
    video_paths = []
    for root, dirs, files in os.walk(args.images):
        for file in files:
            if file.endswith(img_exts):
                image_paths.append(os.path.join(root, file))
            elif file.endswith(video_exts):
                video_paths.append(os.path.join(root, file))

    image_paths.sort()
    video_paths.sort()

    # Extract the EXIF data
    image_data = {}
    with exiftool.ExifToolHelper() as et:
        for image_path in image_paths:
            md5 = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
            exif_tags = et.get_metadata(image_path)[0]
            image_name = os.path.basename(image_path)

            image_data[image_name] = {}
            image_data[image_name]["orig_path"] = image_path
            image_data[image_name]["md5"] = md5
            image_data[image_name]["exif_tags"] = exif_tags

            # Get the sensor name
            sensor_name, sensor_info = exif.exif_to_sensor_info(
                exif_tags, args.image_res_ratio
            )
            if sensor_name not in sensor_list:
                sensor_list[sensor_name] = sensor_info
                print("- add new sensor {} to the sensor list".format(sensor_name))
            image_data[image_name]["sensor_name"] = sensor_name

            image_data[image_name]["orig_width"] = exif_tags["EXIF:ExifImageWidth"]
            image_data[image_name]["orig_height"] = exif_tags["EXIF:ExifImageHeight"]

            # Get the capture time
            image_data[image_name]["capture_time"] = exif.exif_to_time(exif_tags)

            # Get new image name
            image_data[image_name]["new_name"] = (
                utils.time_to_name(image_data[image_name]["capture_time"])
                + os.path.splitext(image_name)[1]
            )

            # Get the GNSS coordinates
            image_data[image_name]["coords_wgs84"] = exif.exif_to_WGS84(exif_tags)

    # Organize the images by date and sensor
    print("- organize the images by date and sensor")
    image_data_org = {}
    for image_name in image_data:
        date_dir = os.path.join(
            args.dataset, image_data[image_name]["capture_time"].strftime("%Y-%m-%d")
        )
        sensor_dir = os.path.join(date_dir, image_data[image_name]["sensor_name"])

        orig_image_path = image_data[image_name]["orig_path"]
        new_image_path = os.path.join(sensor_dir, image_data[image_name]["new_name"])
        new_image_relpath = os.path.relpath(new_image_path, args.dataset)

        if new_image_relpath in image_database_relpaths:
            print("  - skip already processed image: {}".format(new_image_relpath))
            continue

        image_data[image_name]["new_path"] = new_image_path

        os.makedirs(sensor_dir, exist_ok=True)
        
        if args.image_res_ratio == 1.0:
            # Just copy the image if not resizing to speed up the process
            shutil.copy2(orig_image_path, new_image_path)
        else:
            utils.copy_resize_image(
                orig_image_path, new_image_path, args.image_res_ratio
            )


        image_data_org[new_image_relpath] = image_data[image_name]

    image_data = image_data_org

    # Load the GPX file
    if args.gpx_file is not None:
        if os.path.isdir(args.gpx_file):
            gpx_files = [
                os.path.join(args.gpx_file, f)
                for f in os.listdir(args.gpx_file)
                if f.endswith(".gpx")
            ]
            gpx_files.sort()
        elif os.path.isfile(args.gpx_file):
            gpx_files = [args.gpx_file]
        else:
            raise FileNotFoundError("The given GPX file does not exist.")
        
        timestamps = []
        coords_wgs84 = []

        for gpx_file in gpx_files:
            print("- load GPX file: {}".format(gpx_file))
            timestamps_curr, coords_wgs84_curr = gpx.load_gpx_file(gpx_file)

            timestamps.append(timestamps_curr)
            coords_wgs84.append(coords_wgs84_curr)

        with exiftool.ExifToolHelper() as et:
            for image_relpath in image_data:
                if args.gpx_overwrite or image_data[image_relpath]["coords_wgs84"] is None:
                    image_data[image_relpath]["coords_wgs84"] = gpx.gpx_interpolate_mult(
                        timestamps,
                        coords_wgs84,
                        image_data[image_relpath]["capture_time"],
                        args.gpx_mode,
                    )
                    et.set_tags(
                        image_data[image_relpath]["new_path"],
                        exif.WGS84_to_exif(
                            image_data[image_relpath]["coords_wgs84"]
                        ),
                        params=["-overwrite_original"],
                    )

    # Load the videos and extract the frames
    for video_path in video_paths:
        video_metadata = video.get_metadata(video_path)

        date_dir = os.path.join(
            args.dataset, video_metadata["start_time"].strftime("%Y-%m-%d")
        )
        sensor_dir = os.path.join(date_dir, video_metadata["sensor_name"])
        os.makedirs(sensor_dir, exist_ok=True)

        frame_data = video.process_video(
            video_path,
            sensor_dir,
            args.video_fps,
            args.gpx_file,
            args.gpx_mode,
            args.video_res_ratio,
        )

        image_data.update(frame_data)

    # Assign tags to images
    print("- assign tags to images")
    if args.geojson_file is not None:
        print("  - load given GeoJSON file: {}".format(args.geojson_file))
        LT = locations.LocationTagger(args.geojson_file)
    for image_relpath in image_data:
        # Tag the images with day-time tag
        if image_data[image_relpath]["coords_wgs84"] is not None:
            image_data[image_relpath]["tag_daytime"] = daytime.get_daytime_tag(
                image_data[image_relpath]["coords_wgs84"][0, 0],
                image_data[image_relpath]["coords_wgs84"][1, 0],
                image_data[image_relpath]["capture_time"],
            )
        else:
            image_data[image_relpath]["tag_daytime"] = "unknown"
        
        # Tag the images with location tag
        if image_data[image_relpath]["coords_wgs84"] is not None and args.geojson_file is not None:
            # There might be multiple locations per image at the places where 
            # multiple location polygons in the GeoJSON file intersect
            image_data[image_relpath]["tag_location"] = LT.tag_points(
                image_data[image_relpath]["coords_wgs84"]
            )[0]

        else:
            image_data[image_relpath]["tag_location"] = ["unknown"]
        
        # TODO: Tag the iamges with other image-based tags

    # Define the metadata for each sensor directory
    print("- define the metadata for each sensor subdirectory")
    for date_dir in os.listdir(args.dataset):
        if not os.path.isdir(os.path.join(args.dataset, date_dir)):
            continue
        for sensor_dir in os.listdir(os.path.join(args.dataset, date_dir)):
            sensor_name = sensor_dir

            # Load the metadata file
            metadata_file = os.path.join(
                args.dataset, date_dir, sensor_dir, "metadata.yaml"
            )
            if os.path.exists(metadata_file):
                metadata = utils.load_metadata(metadata_file)
                metadata["image_list"] = []
                metadata["seq_id"] = 0
                for sname, seq in metadata["sequences"].items():
                    metadata["image_list"].extend(seq["images"].keys())
                    metadata["seq_id"] = max(
                        int(sname.split("-")[1]), metadata["seq_id"]
                    )
                metadata_image_relpaths = metadata["image_list"].copy()
            else:
                metadata = {}
                metadata["sequences"] = {}
                metadata["image_list"] = []
                metadata["seq_id"] = 0
                metadata_image_relpaths = []

            sensor_image_relpaths = os.listdir(
                os.path.join(args.dataset, date_dir, sensor_dir)
            )
            sensor_image_relpaths = [os.path.join(date_dir, sensor_dir, p) for p in sensor_image_relpaths]
            sensor_image_relpaths = [p for p in sensor_image_relpaths if p not in image_database_relpaths]
            sensor_image_relpaths = [p for p in sensor_image_relpaths if p.endswith(img_exts)]
            sensor_image_relpaths.sort()

            # Divide the images into sequences
            seq = {}
            seq_id = metadata["seq_id"] + 1
            seq_name = "seq-{:0>3d}".format(seq_id)
            seq["images"] = {}
            seq["coordinates"] = np.empty((3, 0))

            for i in range(len(sensor_image_relpaths)):
                curr_image_relpath = sensor_image_relpaths[i]
                prev_image_relpath = sensor_image_relpaths[i - 1]

                # Skip the images that are already in the loaded metadata file
                if curr_image_relpath in metadata["image_list"]:
                    continue

                # Initialize new sequence
                if len(seq["images"]) == 0:
                    seq["images"][curr_image_relpath] = {}
                    image_data[curr_image_relpath]["sequence"] = seq_name

                    if image_data[curr_image_relpath]["coords_wgs84"] is not None:
                        seq["coordinates"] = np.append(
                            seq["coordinates"],
                            image_data[curr_image_relpath]["coords_wgs84"],
                            axis=1,
                        )
                    continue

                # Compute time difference to the last image
                time_diff = (
                    image_data[curr_image_relpath]["capture_time"]
                    - image_data[prev_image_relpath]["capture_time"]
                ).total_seconds()

                # Compute space distance to all the images in the sequence
                if image_data[curr_image_relpath]["coords_wgs84"] is not None:
                    space_dist = float(
                        np.min(
                            gnss_tools.dist_haversine(
                                gnss_tools.prep_coords_wgs84(
                                    image_data[curr_image_relpath]["coords_wgs84"]
                                ),
                                gnss_tools.prep_coords_wgs84(seq["coordinates"]),
                            )
                        )
                    )
                else:
                    space_dist = 0.0

                if time_diff <= args.seq_max_td and space_dist <= args.seq_max_dist:
                    seq["images"][curr_image_relpath] = {}
                    image_data[curr_image_relpath]["sequence"] = seq_name

                    if image_data[curr_image_relpath]["coords_wgs84"] is not None:
                        seq["coordinates"] = np.append(
                            seq["coordinates"],
                            image_data[curr_image_relpath]["coords_wgs84"],
                            axis=1,
                        )
                else:
                    del seq["coordinates"]
                    metadata["sequences"][seq_name] = seq

                    seq = {}
                    seq_id += 1
                    seq_name = "seq-{:0>3d}".format(seq_id)
                    seq["images"] = {curr_image_relpath: {}}
                    image_data[curr_image_relpath]["sequence"] = seq_name

                    if image_data[curr_image_relpath]["coords_wgs84"] is not None:
                        seq["coordinates"] = np.array(
                            image_data[curr_image_relpath]["coords_wgs84"]
                        )
                    else:
                        seq["coordinates"] = np.empty((3, 0))
                        
                    seq["inter_sequence_time_diff"] = time_diff
                    seq["inter_sequence_space_dist"] = space_dist

            del seq["coordinates"]

            if len(seq["images"]) > 0:
                metadata["sequences"][seq_name] = seq

            # Add metadata to each image in the sequence
            for seq in metadata["sequences"].values():
                for image_relpath in seq["images"].keys():
                    if image_relpath in metadata_image_relpaths:
                        continue

                    seq["images"][image_relpath]["capture_time"] = image_data[image_relpath][
                        "capture_time"
                    ]
                    seq["images"][image_relpath]["coords_wgs84"] = image_data[image_relpath][
                        "coords_wgs84"
                    ]
                    seq["images"][image_relpath]["orig_width"] = image_data[image_relpath][
                        "orig_width"
                    ]
                    seq["images"][image_relpath]["orig_height"] = image_data[image_relpath][
                        "orig_height"
                    ]
                    seq["images"][image_relpath]["tag_location"] = image_data[image_relpath][
                        "tag_location"
                    ]
                    seq["images"][image_relpath]["tag_daytime"] = image_data[image_relpath][
                        "tag_daytime"
                    ]

                    # TODO: Add weather tag

                # Tag the sequences based on the tags of the images
                seq["tag_location"], seq["tag_daytime"] = utils.tag_sequence(seq["images"])

            del metadata["image_list"]
            del metadata["seq_id"]

            metadata["date"] = date_dir
            metadata["sensor"] = sensor_name

            print("  - write metadata file: {}".format(os.path.relpath(metadata_file, args.dataset)))
            utils.save_metadata(metadata_file, metadata)

    # Save the sensor list
    if args.new_sensor_file:
        with open(args.new_sensor_file, "wt") as f:
            yaml.dump(sensor_list, f)
    else:
        with open(args.sensor_file, "wt") as f:
            yaml.dump(sensor_list, f)
    
    # Save the image database
    new_image_database = ImageDatabase(root_path = args.dataset, db = image_data)
    if args.new_image_database_file:
        new_image_database.save(args.new_image_database_file)
    else:
        new_image_database.save(args.image_database_file)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.image_res_ratio < 0.0 or args.image_res_ratio > 1.0:
        raise ValueError(
            "The resolution ratio must be in the (0.0 - 1.0) range."
            )
    
    if args.video_res_ratio < 0.0 or args.video_res_ratio > 1.0:
        raise ValueError(
            "The resolution ratio must be in the (0.0 - 1.0) range."
            )

    main(args)

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
import datetime
import datetime

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
    help="The YAML file with the sensor list updated based on the new images.",
)
parser.add_argument(
    "--seq_max_td",
    type=float,
    default=30.0,
    help="The maximum time difference between two subsequent images in seconds to consider them as part of the same sequence.",
)
parser.add_argument(
    "--seq_max_dist",
    type=float,
    default=30.0,
    help="The maximum distance for two images in meters to consider them as part of the same sequence.",
)
parser.add_argument(
    "--gpx_file", 
    type=str, 
    help="The GPX file with the GPS data.",
)
parser.add_argument(
    "--gpx_mode",
    type=str,
    choices=["nearest", "nearest_outside", "linear", "linear_outside"],
    default="nearest",
    help="GPX interpolation mode. The \"nearest\" and \"linear\" raise an error if the time is outside the GPX time range.",
)
parser.add_argument(
    "--video_fps",
    type=float,
    default=2.0,
    help="The frame rate for the video frame extraction.",
)


img_exts = (".jpg", ".jpeg", ".png")
video_exts = ".mp4"


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
            exif_tags = et.get_metadata(image_path)[0]
            image_name = os.path.basename(image_path)

            image_data[image_name] = {}
            image_data[image_name]["orig_path"] = image_path
            image_data[image_name]["exif_tags"] = exif_tags

            # Get the sensor name
            sensor_name, sensor_info = exif.exif_to_sensor_info(exif_tags)
            if sensor_name not in sensor_list:
                sensor_list[sensor_name] = sensor_info
                print("- add new sensor {} to the sensor list".format(sensor_name))
            image_data[image_name]["sensor_name"] = sensor_name

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
        image_data[image_name]["new_path"] = new_image_path

        os.makedirs(sensor_dir, exist_ok=True)
        shutil.copy(orig_image_path, new_image_path)

        image_data_org[image_data[image_name]["new_name"]] = image_data[image_name]

    image_data = image_data_org

    # Load the GPX file
    if args.gpx_file:
        print("- load GPX file: {}".format(args.gpx_file))
        timestamps, coords_wgs84 = gpx.load_gpx_file(args.gpx_file)

        for image_name in image_data:
            if image_data[image_name]["coords_wgs84"] is None:
                image_data[image_name]["coords_wgs84"] = gpx.gpx_interpolate(
                    timestamps,
                    coords_wgs84,
                    image_data[image_name]["capture_time"],
                    args.gpx_mode,
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
            video_path, sensor_dir, args.video_fps, args.gpx_file, args.gpx_mode
        )

        image_data.update(frame_data)

    # Assign tags to images
    print("- assign tags to images")
    if args.geojson_file:
        print("  - load given GeoJSON file: {}".format(args.geojson_file))
        LT = locations.LocationTagger(args.geojson_file)
    for image_name in image_data:
        image_data[image_name]["tag_daytime"] = daytime.get_daytime_tag(
            image_data[image_name]["coords_wgs84"][0, 0],
            image_data[image_name]["coords_wgs84"][1, 0],
            image_data[image_name]["capture_time"],
        )
        if args.geojson_file:
            # There might be multiple locations per image if multiple location
            # polygons in the GeoJSON file intersect
            image_data[image_name]["tag_location"] = LT.tag_points(
                image_data[image_name]["coords_wgs84"]
            )[0]
        # TODO: Add weather tag

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
            else:
                metadata = {}
                metadata["sequences"] = {}
                metadata["image_list"] = []
                metadata["seq_id"] = 0

            sensor_image_names = os.listdir(
                os.path.join(args.dataset, date_dir, sensor_dir)
            )
            sensor_image_names = [p for p in sensor_image_names if p.endswith(img_exts)]
            sensor_image_names.sort()

            # Divide the images into sequences
            seq = {}
            seq_id = metadata["seq_id"] + 1
            seq["images"] = {}
            seq["coordinates"] = np.empty((3, 0))

            for i in range(len(sensor_image_names)):
                # Skip the images that are already in the loaded metadata file
                if sensor_image_names[i] in metadata["image_list"]:
                    continue

                # Initialize new sequence
                if len(seq["images"]) == 0:
                    seq["images"][os.path.basename(sensor_image_names[i])] = {}
                    seq["coordinates"] = np.append(
                        seq["coordinates"],
                        image_data[sensor_image_names[i]]["coords_wgs84"],
                        axis=1,
                    )
                    continue

                # Compute time difference to the last image
                time_diff = (
                    image_data[sensor_image_names[i]]["capture_time"]
                    - image_data[sensor_image_names[i - 1]]["capture_time"]
                ).total_seconds()

                # Compute space distance to all the images in the sequence
                space_dist = float(
                    np.min(
                        gnss_tools.dist_haversine(
                            gnss_tools.prep_coords_wgs84(
                                image_data[sensor_image_names[i]]["coords_wgs84"]
                            ),
                            gnss_tools.prep_coords_wgs84(seq["coordinates"]),
                        )
                    )
                )

                if time_diff <= args.seq_max_td and space_dist <= args.seq_max_dist:
                    seq["images"][os.path.basename(sensor_image_names[i])] = {}
                    seq["coordinates"] = np.append(
                        seq["coordinates"],
                        image_data[sensor_image_names[i]]["coords_wgs84"],
                        axis=1,
                    )
                else:
                    del seq["coordinates"]
                    seq_name = "seq-{:0>3d}".format(seq_id)
                    metadata["sequences"][seq_name] = seq

                    seq = {}
                    seq_id += 1
                    seq["images"] = {os.path.basename(sensor_image_names[i]): {}}
                    seq["coordinates"] = np.array(
                        image_data[sensor_image_names[i]]["coords_wgs84"]
                    )
                    seq["inter_sequence_time_diff"] = time_diff
                    seq["inter_sequence_space_dist"] = space_dist

            del seq["coordinates"]

            if len(seq["images"]) > 0:
                seq_name = "seq-{:0>3d}".format(seq_id)
                metadata["sequences"][seq_name] = seq

            # Add metadata to each image in the sequence
            for seq in metadata["sequences"].values():
                for image_path in seq["images"].keys():
                    seq["images"][image_path]["capture_time"] = image_data[image_path][
                        "capture_time"
                    ]
                    seq["images"][image_path]["coords_wgs84"] = image_data[image_path][
                        "coords_wgs84"
                    ]
                    seq["images"][image_path]["tag_location"] = image_data[image_path][
                        "tag_location"
                    ]
                    seq["images"][image_path]["tag_daytime"] = image_data[image_path][
                        "tag_daytime"
                    ]
                    # TODO: Add weather tag

                # Tag the sequences based on the tags of the images
                seq["tag_location"], seq["tag_daytime"] = utils.tag_sequence(seq["images"])

            del metadata["image_list"]
            del metadata["seq_id"]

            metadata["date"] = date_dir
            metadata["sensor"] = sensor_name

            print("  - write metadata file: {}".format(metadata_file))
            utils.save_metadata(metadata_file, metadata)

    # Save the sensor list
    if args.new_sensor_file:
        with open(args.new_sensor_file, "wt") as f:
            yaml.dump(sensor_list, f)
    else:
        with open(args.sensor_file, "wt") as f:
            yaml.dump(sensor_list, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

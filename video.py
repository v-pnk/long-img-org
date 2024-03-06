#!/usr/bin/env python3

"""
Extract frames with valid metadata from a video file.
"""

import argparse
import cv2
import exiftool
import os
import datetime

import gpx
import utils
import exif


parser = argparse.ArgumentParser(
    description="Extract frames with valid metadata from a video file."
)
parser.add_argument(
    "video", 
    type=str, 
    help="The input video file path."
)
parser.add_argument(
    "output", 
    type=str, 
    help="Path to the output directory for the frames."
)
parser.add_argument(
    "-r", "--frame_rate", 
    type=int, 
    default=1.0, 
    help="Frame rate for the extraction (frames per second)."
)
parser.add_argument(
    "--gpx_file", 
    type=str, 
    help="Path to the GPX track file / directory with files."
)
parser.add_argument(
    "--gpx_mode", 
    type=str,
    choices=["nearest", "nearest_outside", "linear", "linear_outside"],
    default="nearest",
    help="GPX interpolation mode. The \"nearest\" and \"linear\" raise an error if the time is outside the GPX time range.",
)
parser.add_argument(
    "--res_ratio",
    type=float,
    default=1.0,
    help="Resolution ratio used for downsampling the frames - can be in (0.0 - 1.0) range."
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print notes on the progress."
)


def process_video(video_path, frame_dir, frame_rate, gpx_file=None, gpx_mode="nearest", res_ratio=1.0, verbose=False):
    """
    Process a video file. Extract frames with valid metadata and optionally
    add GPS data to the frames.

    Parameters:
    video_path (str): Video file path.
    frame_dir (str): Output directory for the frames.
    frame_rate (float): Frame rate for the extraction (frames per second).
    gpx_file (str): GPX file path / path to directory with files.

    """

    if verbose:
        print(f"- processing video: {video_path}")
        print(f"- loading video metadata")
    metadata = get_metadata(video_path)

    if verbose:
        print(f"- extracting frames from the video")
    frames = extract_frames(video_path, metadata["start_time"], frame_rate, res_ratio)

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
            if verbose:
                print(f"- loading GPS data from: {gpx_file}")
            
            timestamps_curr, coords_wgs84_curr = gpx.load_gpx_file(gpx_file)

            timestamps.append(timestamps, timestamps_curr)
            coords_wgs84.append(coords_wgs84, coords_wgs84_curr)

        
        for frame in frames:
            frame["coords_wgs84"] = gpx.gpx_interpolate_mult(
                timestamps, coords_wgs84, frame["capture_time"], mode=gpx_mode
            )
            frame["sensor_name"] = metadata["sensor_name"]

    image_data = {}

    if verbose:
        print(f"- saving frames to: {frame_dir}")

    with exiftool.ExifToolHelper() as et:
        for frame in frames:
            image_name = utils.time_to_name(frame["capture_time"])
            image_path = os.path.join(frame_dir, image_name + ".jpg")
            cv2.imwrite(image_path, frame["image"])
            del frame["image"]

            exif_tags = {
                    "XResolution": metadata["x_resolution"],
                    "YResolution": metadata["y_resolution"],
                    "ImageWidth": round(metadata["width"] * res_ratio),
                    "ImageHeight": round(metadata["height"] * res_ratio),
                }
            exif_tags.update(exif.time_to_exif(frame["capture_time"]))
            if frame["coords_wgs84"] is not None:
                exif_tags.update(exif.WGS84_to_exif(frame["coords_wgs84"]))
                
            et.set_tags(image_path, exif_tags, params=["-overwrite_original"])

            image_data[image_name] = frame

    return image_data


def extract_frames(video_path, start_time, frame_rate=1, res_ratio=1.0):
    """
    Extract frames with valid capture times from a video file.

    Parameters:
    video (str): Video file.
    start_time (datetime.datetime): The video capture start date and time.
    frame_rate (float): Frame rate for the extraction (frames per second).


    Returns:
    frames (list): List of frames with valid capture times.
    """

    frames = []

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_rate = min(cap.get(cv2.CAP_PROP_FPS), frame_rate)

    ideal_capture_time = 0.0
    success, frame = cap.read()
    while success:
        this_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        next_time = this_time + 1.0 / cap.get(cv2.CAP_PROP_FPS)

        if abs(this_time - ideal_capture_time) < abs(next_time - ideal_capture_time):
            data = {}

            # Downsample the frame
            if res_ratio < 1.0:
                frame_resized = cv2.resize(
                    frame, (0, 0), fx=res_ratio, fy=res_ratio, interpolation=cv2.INTER_AREA
                )

            data["image"] = frame_resized
            data["capture_time"] = start_time + datetime.timedelta(seconds=this_time)
            data["orig_width"] = frame.shape[1]
            data["orig_height"] = frame.shape[0]
            
            frames.append(data)
            ideal_capture_time += 1.0 / frame_rate

        success, frame = cap.read()

    cap.release()

    return frames


def get_metadata(video_path):
    """
    Get selected metadata of a video file.

    Parameters:
    video_path (str): Video file path.

    Returns:
    metadata (dict): Selected metadata of the video file.
    """

    metadata = {}

    with exiftool.ExifToolHelper() as et:
        exif_metadata = et.get_metadata(video_path)[0]

    # Print all the available video metadata
    # for tag, val in exif_metadata.items():
    #     print(f"{tag}: {val}")

    # There is no tag for the video capture time, which would include time
    # zone information --> all the times are in UTC / GMT.
    video_end_time = datetime.datetime.strptime(
        exif_metadata["QuickTime:CreateDate"], "%Y:%m:%d %H:%M:%S"
    )
    video_end_time = video_end_time.replace(tzinfo=datetime.timezone.utc)
    video_duration = datetime.timedelta(seconds=exif_metadata["QuickTime:Duration"])
    video_start_time = video_end_time - video_duration

    metadata["start_time"] = video_start_time
    metadata["width"] = exif_metadata["QuickTime:ImageWidth"]
    metadata["height"] = exif_metadata["QuickTime:ImageHeight"]
    metadata["x_resolution"] = exif_metadata["QuickTime:XResolution"]
    metadata["y_resolution"] = exif_metadata["QuickTime:YResolution"]
    metadata["bit_depth"] = exif_metadata["QuickTime:BitDepth"]

    # Put together some sensor name from the QuickTime tags
    # - the manufacturer or model tags are not always present
    # - QuickTime tags do not contain the focal length
    # - sometimes the manufacturer and model tags are in the author tag
    sensor_name = ""
    if "QuickTime:Make" in exif_metadata:
        sensor_name += exif_metadata["QuickTime:Make"] + "_"
    if "QuickTime:Model" in exif_metadata:
        sensor_name += exif_metadata["QuickTime:Model"] + "_"

    if not sensor_name and "QuickTime:Author" in exif_metadata:
        sensor_name = exif_metadata["QuickTime:Author"] + "_"

    if not sensor_name:
        sensor_name = "video_"

    sensor_name = sensor_name.replace(" ", "-")
    
    sensor_name += f"{metadata['width']}x{metadata['height']}_VF"

    metadata["sensor_name"] = sensor_name

    return metadata


if __name__ == "__main__":
    args = parser.parse_args()

    if args.res_ratio < 0.0 or args.res_ratio > 1.0:
        raise ValueError("The resolution ratio must be in the (0.0 - 1.0) range.")

    process_video(
        args.video, 
        args.output, 
        args.frame_rate, 
        args.gpx_file, 
        args.gpx_mode, 
        args.res_ratio,
        verbose=True
    )

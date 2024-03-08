"""
Image database class implementation and query command-line tool. 
The command-line tool filters the input image database based on a query string,
which specifies the filter conditions.

The query string has the format: 
"key1 : value1; key2 : value2"
The valueN can be a single value, a range between two values ("[min, max]"), 
an open range ("[min" / "max]"), or a list of values ("val1, val2, val3"). 

The valid keys are:
- capture_date
- capture_year
- capture_month
- capture_hour
- sensor_name
- tag_location
- tag_daytime
- latitude
- longitude
- altitude

The command-line tool can also save the filtered database to a file,
create a file containing the list of images in the filtered database, and
copy the images in the filtered database to a destination directory.

"""


import os
import shutil
import datetime
import csv
import argparse

import numpy as np


parser = argparse.ArgumentParser(description="Query image database.")
parser.add_argument(
    "db_path", 
    type=str, 
    help="The path to the image database file."
)
parser.add_argument(
    "query", 
    type=str, 
    help="The query string."
)
parser.add_argument(
    "--dataset_root", 
    type=str, 
    help="The root path of the dataset. Directory of the database file by default."
)
parser.add_argument(
    "--db_out", 
    type=str, 
    help="Save the filtered database."
)
parser.add_argument(
    "--img_list", 
    type=str, 
    help="Save the list of images in the filtered database."
)
parser.add_argument(
    "--img_list_mode", 
    type=str, 
    default="relpath", 
    choices=["relpath", "abspath", "name"],
    help="The mode of the naming of the images in the list file."
)
parser.add_argument(
    "--img_dir", 
    type=str,
    help="Copy the images in the filtered database to a destination directory."
)
parser.add_argument(
    "--img_dir_mode", 
    type=str, 
    default="relpath_to_name", 
    choices=["relpath_to_name", "name", "keep_relpaths"], 
    help="The mode of the naming of the copied images."
)


def main(args):
    dataset_root = os.path.dirname(args.db_path) if args.dataset_root is None else args.dataset_root
    db_orig = ImageDatabase(root_path=dataset_root)
    db_orig.load(args.db_path)
    db_filt = db_orig.filter(args.query)

    if args.db_out is not None:
        db_filt.save(args.db_out)
    
    if args.img_list is not None:
        db_filt.create_image_list_file(args.img_list, mode=args.img_list_mode)
    
    if args.img_dir is not None:
        db_filt.copy_images(args.img_dir, mode=args.img_dir_mode)


class ImageDatabase:
    """Class for the image database."""

    # Valid metadata fields
    metadata_fields = [
        "capture_time",
        "sensor_name",
        "sequence",
        "tag_location",
        "tag_daytime",
        "coords_wgs84",
        "orig_width",
        "orig_height"
        "orig_relpath",
    ]

    # Valid filter keys
    filter_keys = [
        "capture_date",
        "capture_year", 
        "capture_month", 
        "capture_hour",
        "sensor_name",
        "tag_location",
        "tag_daytime",
        "latitude", 
        "longitude", 
        "altitude",
    ]

    filter_key_to_metadata_key = {
        "capture_date": "capture_time",
        "capture_year": "capture_time",
        "capture_month": "capture_time",
        "capture_hour": "capture_time",
        "sensor_name": "sensor_name",
        "tag_location": "tag_location",
        "tag_daytime": "tag_daytime",
        "latitude": "coords_wgs84",
        "longitude": "coords_wgs84",
        "altitude": "coords_wgs84",
    }

    def __init__(self, db=None, root_path=None):
        """Initialize the image database."""

        if db is None:
            db = {}
        
        self.db = db
        self.root_path = root_path


    def __len__(self):
        """Get the number of images in the database.

        Returns:
        int: The number of images.

        """

        return len(self.db)


    def __getitem__(self, img_relpath):
        """Get the metadata of an image.

        Parameters:
        img_relpath (str): The relative path of the image.

        Returns:
        dict: The metadata of the image.

        """

        return self.db[img_relpath]


    def __setitem__(self, img_relpath, img_data):
        """Set the metadata of an image. Check if the metadata fields are valid.

        Parameters:
        img_relpath (str): The relative path of the image.
        img_data (dict): The metadata of the image.

        """

        for key in img_data:
            assert key in self.metadata_fields, "Invalid metadata field: " + key

        self.db[img_relpath] = img_data


    def __delitem__(self, img_relpath):
        """Delete an image from the database.

        Parameters:
        img_relpath (str): The relative path of the image.

        """

        del self.db[img_relpath]


    def __iter__(self):
        """Get an iterator for the image database.

        Returns:
        iter: The iterator.

        """

        return iter(self.db)


    def __next__(self):
        """Get the next image in the database.

        Returns:
        str: The relative path of the next image.

        """

        return next(self.db)


    def __contains__(self, img_relpath):
        """Check if an image is in the database.

        Parameters:
        img_relpath (str): The relative path of the image.

        Returns:
        bool: True if the image is in the database, False otherwise.

        """

        return img_relpath in self.db


    def load(self, image_database_path):
        """Load the image database from a file.

        Parameters:
        image_database_path (str): The path to the image database file.

        """

        assert os.path.exists(image_database_path), "The given image database file does not exist: " + image_database_path

        with open(image_database_path, 'rt', newline='') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                img_name = row[0]
                date = datetime.datetime.strptime(row[1] + " " + row[2] + " " + row[3], "%Y-%m-%d %H:%M:%S.%f %z")
                sensor_name = row[4]
                sequence = row[5]
                tag_location = row[6]
                tag_daytime = row[7]
                if row[8] == "-":
                    coords_wgs84 = None
                else:
                    latitude = float(row[8])
                    longitude = float(row[9])
                    altitude = float(row[10])
                    coords_wgs84 = np.array([latitude, longitude, altitude]).reshape(3, 1)
                orig_width = int(row[11])
                orig_height = int(row[12])

                img_relpath = os.path.join(date.strftime("%Y-%m-%d"), 
                                           sensor_name, img_name)

                self.db[img_relpath] = {
                    "capture_time": date,
                    "sensor_name": sensor_name,
                    "sequence": sequence,
                    "tag_location": tag_location,
                    "tag_daytime": tag_daytime,
                    "coords_wgs84": coords_wgs84,
                    "orig_width": orig_width,
                    "orig_height": orig_height
                }


    def save(self, image_database_path):
        """Save the image database to a file.

        Parameters:
        image_database_path (str): The path to the image database file.

        """

        if os.path.exists(image_database_path):
            file_mode = "at"
        else:
            file_mode = "wt"

        with open(image_database_path, mode=file_mode, newline='') as f:
            csv_writer = csv.writer(f)

            image_relpath_list = list(self.db.keys())
            image_relpath_list.sort()

            for img_relpath in image_relpath_list:
                img_name = os.path.basename(img_relpath)
                img_data = self.db[img_relpath]
                date = img_data["capture_time"].strftime("%Y-%m-%d")
                time = img_data["capture_time"].strftime("%H:%M:%S.%f")[:-3]
                timezone = img_data["capture_time"].strftime("%z")
                sensor_name = img_data["sensor_name"]
                sequence = img_data["sequence"]
                tag_location = img_data["tag_location"]
                tag_daytime = img_data["tag_daytime"]
                if "coords_wgs84" not in img_data or img_data["coords_wgs84"] is None:
                    latitude = "-"
                    longitude = "-"
                    altitude = "-"
                else:
                    latitude = "{:.8f}".format(img_data["coords_wgs84"][0,0])
                    longitude = "{:.8f}".format(img_data["coords_wgs84"][1,0])
                    altitude = "{:.4f}".format(img_data["coords_wgs84"][2,0])
                orig_width = img_data["orig_width"]
                orig_height = img_data["orig_height"]

                row_data = [img_name, 
                            date, 
                            time, 
                            timezone, 
                            sensor_name, 
                            sequence, 
                            tag_location, 
                            tag_daytime, 
                            latitude, 
                            longitude, 
                            altitude, 
                            orig_width, 
                            orig_height]
                csv_writer.writerow(row_data)


    def get_relpaths(self):
        """Get the relative paths of the images in the database.

        Returns:
        list: The list of relative paths.

        """

        rel_paths = list(self.db.keys())
        rel_paths.sort()
        return rel_paths


    def add_image(self, image_name, data):
        """Add an image to the database.

        Parameters:
        image_name (str): The image name.
        metadata (dict): The metadata of the image.

        """

        self.db[image_name] = data


    def filter(self, filter_str):
        """Filter the image database based on the given string, returning 
        a new database.
        
        Parameters:
        filter_str (str): The filter string with format:
        "key1 : value1; key2 : value2", where the valueN can be a single value, 
        a range between two values ("[min, max]"), an open range ("[min" / "max]"),
        or a list of values ("val1, val2, val3"). 

        Returns:
        ImageDatabase: The filtered image database.

        """

        filter_str = filter_str.lower()
        filter_dict = {}
        for substr in filter_str.split(";"):
            key, value = substr.strip().split(":")
            key = key.strip()
            value = value.strip()

            assert key in self.filter_keys, "Invalid key in the filter string: " + key
            
            if value.startswith("[") and value.endswith("]"):
                mode = "range"
                left_limit, right_limit = value[1:-1].split(",")
                left_limit = left_limit.strip()
                right_limit = right_limit.strip()
                filter_values = [left_limit, right_limit]
            elif value.startswith("["):
                mode = "lrange"
                left_limit = value[1:].strip()
                filter_values = [left_limit]
            elif value.endswith("]"):
                mode = "rrange"
                right_limit = value[:-1].strip()
                filter_values = [right_limit]
            elif "," in value:
                mode = "list"
                filter_values = [val.strip() for val in value.split(",")]
            else:
                mode = "single"
                filter_values = [value]
            
            filter_dict[key] = {"mode": mode, "values": filter_values}

        filtered_db = ImageDatabase(root_path=self.root_path)

        for img_name, img_data in self.db.items():
            passed = True

            for key, val_dict in filter_dict.items():
                metadata_key = self.filter_key_to_metadata_key[key]
                # If the key is not in the image metadata, the image does not 
                # pass the filter
                if metadata_key not in img_data:
                    print("WARN: Key \"{}\" not in image {} metadata".format(key, img_name))
                    passed = False
                    break
                
                filter_mode = val_dict["mode"]
                filter_values = val_dict["values"]

                if key == "capture_date":
                    image_value = img_data[metadata_key].strftime("%Y-%m-%d")
                    filter_values = [datetime.datetime.strptime(val, "%Y-%m-%d") for val in filter_values]

                elif key == "capture_year":
                    image_value = int(img_data[metadata_key].strftime("%Y"))
                    filter_values = [int(val) for val in filter_values]
                
                elif key == "capture_month":
                    image_value = int(img_data[metadata_key].strftime("%m"))
                    filter_values = [int(val) for val in filter_values]

                elif key == "capture_hour":
                    image_value = int(img_data[metadata_key].strftime("%H"))
                    filter_values = [int(val) for val in filter_values]

                elif key in ["sensor_name"]:
                    image_value = img_data[metadata_key]
                    
                elif key in ["tag_location", "tag_daytime"]:
                    image_value = img_data[metadata_key]
                
                elif key == "latitude":
                    image_value = img_data[metadata_key][0,0]

                    for val in filter_values:
                        if val.lower().endswith("n"):
                            val = float(val[:-1])
                        elif val.lower().endswith("s"):
                            val = -float(val[:-1])
                        else:
                            val = float(val)

                    filter_values = [float(val) for val in filter_values]

                elif key == "longitude":
                    image_value = img_data[metadata_key][1,0]

                    for val in filter_values:
                        if val.lower().endswith("e"):
                            val = float(val[:-1])
                        elif val.lower().endswith("w"):
                            val = -float(val[:-1])
                        else:
                            val = float(val)

                    filter_values = [float(val) for val in filter_values]

                elif key == "altitude":
                    image_value = img_data[metadata_key][2,0]
                    filter_values = [float(val) for val in filter_values]
                
                cyclic_value = key in ["capture_hour", "capture_month", "capture_year", "longitude"]

                passed = passed and compare_metadata(image_value, filter_values, filter_mode, cyclic=cyclic_value)

                if not passed:
                    break

            if passed:
                filtered_db.add_image(img_name, img_data)

        return filtered_db


    def create_image_list_file(self, image_list_path, mode="relpath"):
        """Create a file containing the list of images in the database.

        Parameters:
        image_list_path (str): The path to the image list file.
        mode (str): The mode of the image list file. Can be "relpath", 
        "abspath", or "name".

        """

        if mode == "abspath":
            assert self.root_path is not None, "The dataset root path is not set."

        with open(image_list_path, "wt") as f:
            for img_relpath in self.db:
                if mode == "relpath":
                    f.write(img_relpath + "\n")
                elif mode == "abspath":
                    f.write(os.path.join(self.root_path, img_relpath) + "\n")
                elif mode == "name":
                    f.write(os.path.basename(img_relpath) + "\n")
                else:
                    assert False, "Invalid mode: " + mode


    def copy_images(self, dest_dir, mode="relpath_to_name"):
        """Copy the images in the database to a destination directory.

        Parameters:
        dest_dir (str): The path to the destination directory.
        mode (str): The mode of the image list file. Can be "relpath_to_name" 
        (all images into single directory and rename them with their original 
        relative paths), "name" (all images into single directory and keep 
        names), or "keep_relpaths" (keep the original directory structure).

        """

        assert self.root_path is not None, "The dataset root path is not set."
        assert os.path.exists(dest_dir), "The given destination directory does not exist: " + dest_dir

        for img_relpath in self.db:
            img_name = os.path.basename(img_relpath)
            if mode == "relpath_to_name":
                img_name = "_".join(img_relpath.split(os.sep))
                img_dest_path = os.path.join(dest_dir, img_name)
            elif mode == "name":
                img_dest_path = os.path.join(dest_dir, img_name)
            elif mode == "keep_relpaths":
                img_dest_path = os.path.join(dest_dir, img_relpath)
            else:
                assert False, "Invalid mode: " + mode

            img_src_path = os.path.join(self.root_path, img_relpath)

            if not os.path.exists(os.path.dirname(img_dest_path)):
                os.makedirs(os.path.dirname(img_dest_path))

            shutil.copy2(img_src_path, img_dest_path)


def compare_metadata(image_value, filter_values, filter_mode, cyclic=False):
    """Compare a metadata value to a filter value.

    Parameters:
    img_val (list[str|int|float]): The metadata value.
    filter_values (list[str|int|float]): The filter value.
    filter_mode (str): The filter mode.
    cyclic (bool): Whether the compared value is cyclic.

    Returns:
    bool: True if the metadata value passes the filter, False otherwise.

    """

    if filter_mode == "single":
        if isinstance(image_value, list):
            return filter_values[0] in image_value
        else:
            return image_value == filter_values[0]
        
    elif filter_mode == "range":
        left_limit, right_limit = filter_values
        if cyclic:
            return compare_range_cyclic(image_value, left_limit, right_limit)
        else:
            return left_limit <= image_value <= right_limit
        
    elif filter_mode == "lrange":
        left_limit = filter_values[0]
        return left_limit <= image_value
    
    elif filter_mode == "rrange":
        right_limit = filter_values[0]
        return image_value <= right_limit
    
    elif filter_mode == "list":
        if isinstance(image_value, list):
            return any(val in filter_values for val in image_value)
        else:
            return image_value in filter_values
        
    else:
        assert False, "Invalid filter mode: " + filter_mode


def compare_range_cyclic(value, left_limit = -float("inf"), right_limit = float("inf")):
    """Compare a value to a range in a cyclic manner.

    Parameters:
    value (float): The value to compare.
    left_limit (float): The left limit of the range.
    right_limit (float): The right limit of the range.

    Returns:
    bool: True if the value is within the range, False otherwise.

    """

    if left_limit < right_limit:
        return left_limit <= value <= right_limit
    else:
        return left_limit <= value or value <= right_limit


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
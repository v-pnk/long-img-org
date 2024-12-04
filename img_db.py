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
import hashlib
import argparse
import copy

import numpy as np

import utils

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
    help="The root path of the input dataset. Using the directory of input database file by default."
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
parser.add_argument(
    "--append_mode",
    type=str,
    default="overwrite",
    choices=["keep_new", "keep_old", "overwrite"],
    help="The mode of the file saving if the there are already existing files in the given paths."
)
parser.add_argument(
    "--subset_size",
    type=int,
    default=-1,
    help="Limit the number of images in the filtered set."
)
parser.add_argument(
    "--subset_mode",
    type=str,
    default="random",
    choices=["random", "regular"],
    help="The mode of the subset selection - sample randomly or regularly."
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="The seed for the random subset selection."
)


def main(args):
    dataset_root = os.path.dirname(args.db_path) if args.dataset_root is None else args.dataset_root
    db_orig = ImageDatabase(root_path=dataset_root)
    db_orig.load(args.db_path)
    db_filt = db_orig.filter(args.query)

    if args.subset_size != -1:
        db_filt = db_filt.get_subset(args.subset_size, mode=args.subset_mode, seed=args.random_seed)

    if args.db_out is not None:
        db_filt.save(args.db_out, args.append_mode)
    
    if args.img_list is not None:
        db_filt.create_image_list_file(args.img_list, img_list_mode=args.img_list_mode, append_mode=args.append_mode)
    
    if args.img_dir is not None:
        db_filt.copy_images(args.img_dir, mode=args.img_dir_mode, append_mode=args.append_mode)


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
        "md5",
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

        assert os.path.isfile(image_database_path), "The given image database file does not exist: " + image_database_path

        with open(image_database_path, 'rt', newline='') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                img_name = row[0]
                date = datetime.datetime.strptime(row[1] + " " + row[2] + " " + row[3], "%Y-%m-%d %H:%M:%S.%f %z")
                sensor_name = row[4]
                sequence = row[5]
                tag_location = str2list(row[6])
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
                md5 = row[13]

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
                    "orig_height": orig_height,
                    "md5": md5,
                }


    def save(self, image_database_path, append_mode="overwrite"):
        """Save the image database to a file.

        Parameters:
        image_database_path (str): The path to the image database file.
        keep (str): The mode of the file saving if the database file already 
        exists. Can be "keep_new" (keep the new data in case of overlap), 
        "keep_old" (keep the old data in case of overlap), or "overwrite" 
        (overwrite all of the old data).

        """

        if os.path.isfile(image_database_path):
            print("- the given image database file already exists")
            if append_mode == "keep_new":
                print("  - keeping the new image data in case of overlap")
                save_db = merge_databases(image_database_path, self, append_mode)
            elif append_mode == "keep_old":
                print("  - keeping the old image data in case of overlap")
                save_db = merge_databases(image_database_path, self, append_mode)
            elif append_mode == "overwrite":
                print("  - overwriting the existing image database file")
                save_db = self
                
        else:
            print("- creating a new image database file")
            save_db = self

        with open(image_database_path, mode="wt", newline='') as f:
            csv_writer = csv.writer(f)

            image_relpath_list = list(save_db.db.keys())
            image_relpath_list.sort()

            for img_relpath in image_relpath_list:
                img_name = os.path.basename(img_relpath)
                img_data = save_db.db[img_relpath]
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
                md5 = img_data["md5"]

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
                            orig_height,
                            md5,
                ]
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

        filter_str = filter_str.strip()
        if len(filter_str) == 0:
            return copy.deepcopy(self)

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
        passed_nums = {key: 0 for key in filter_dict}

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

                passed_this = compare_metadata(image_value, filter_values, filter_mode, cyclic=cyclic_value)
                passed_nums[key] += passed_this
                passed = passed and passed_this

            if passed:
                filtered_db.add_image(img_name, img_data)

        print("- number of images within each category:")
        for key, num in passed_nums.items():
            print("  - {}: {} images".format(key, num))
        print("  ----------------------------------------")
        print("  - total retrieved images: {}".format(len(filtered_db)))

        return filtered_db


    def create_image_list_file(self, image_list_path, img_list_mode="relpath", append_mode="overwrite"):
        """Create a file containing the list of images in the database.

        Parameters:
        image_list_path (str): The path to the image list file.
        mode (str): The mode of the image list file. Can be "relpath", 
        "abspath", or "name".

        """

        if img_list_mode == "abspath":
            assert self.root_path is not None, "The dataset root path is not set."

        if img_list_mode == "relpath":
            list_new = self.get_relpaths()
        elif img_list_mode == "abspath":
            list_new = [os.path.join(save_db.root_path, img_relpath) for img_relpath in self.get_relpaths()]
        elif img_list_mode == "name":
            list_new = [os.path.basename(img_relpath) for img_relpath in self.get_relpaths()]
        else:
            assert False, "Invalid mode: " + img_list_mode

        if os.path.isfile(image_list_path):
            print("- the given image list file already exists")
            if append_mode in ["keep_new", "keep_old"]:
                print("  - merging the new image list to the existing one")
                list_save = merge_img_list(image_list_path, list_new, append_mode)
            elif append_mode == "overwrite":
                print("  - overwriting the existing image list file")
                list_save = list_new
        else:
            print("- creating a new image list file")
            list_save = list_new
        
        list_save.sort()
        
        with open(image_list_path, "wt") as f:
            for img_relpath in list_save:
                    f.write(img_relpath + "\n")


    def copy_images(self, dest_dir, mode="relpath_to_name", append_mode="overwrite"):
        """Copy the images in the database to a destination directory.

        Parameters:
        dest_dir (str): The path to the destination directory.
        mode (str): The mode of the image list file. Can be "relpath_to_name" 
        (all images into single directory and rename them with their original 
        relative paths), "name" (all images into single directory and keep 
        names), or "keep_relpaths" (keep the original directory structure).
        append_mode (str): The mode of the image saving if the there are 
        already existing images in the given directory. Can be "keep_new" (keep 
        the new images in case of overlap), "keep_old" (keep the old images 
        in case of overlap), or "overwrite" (delete all the old images and save 
        the new ones).

        """

        assert self.root_path is not None, "The dataset root path is not set."
        assert os.path.isdir(dest_dir), "The given destination directory does not exist: " + dest_dir

        print("- copying the images to the destination directory")
        progress = utils.rich_progress_bar()
        progress.start()

        if append_mode == "overwrite":
            print("- clearing the destination directory")
            shutil.rmtree(dest_dir)
            os.makedirs(dest_dir)

        task = progress.add_task("  ", total=len(self.db))
        for img_relpath in self.db:
            progress.update(task, advance=1)
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

            if not os.path.isdir(os.path.dirname(img_dest_path)):
                os.makedirs(os.path.dirname(img_dest_path))
            
            if not os.path.isfile(img_src_path):
                print("WARN: Source image not found: " + img_src_path)
                continue

            md5_db = self.db[img_relpath]["md5"]
            md5_src = hashlib.md5(open(img_src_path, 'rb').read()).hexdigest()

            if os.path.isfile(img_dest_path) and append_mode == "keep_old":
                continue
            
            shutil.copy2(img_src_path, img_dest_path)

            md5_dest = hashlib.md5(open(img_dest_path, 'rb').read()).hexdigest()
            
            if md5_db != md5_src:
                print("WARN: MD5 hash of the source image does not match the one in the database - check integrity of the file: " + img_src_path)
            elif md5_db != md5_dest:
                print("WARN: MD5 hash of the copied image does not match the one in the database (the source seems OK) - check integrity of the copied file: " + img_dest_path)

        progress.remove_task(task)
        progress.stop()


    def get_subset(self, subset_size, mode, seed):
        if subset_size == -1:
            return copy.deepcopy(self)
        
        if subset_size >= len(self):
            print("The subset size is greater or equal to the size of the full retrieved set - returning the whole set.")
            return copy.deepcopy(self)

        subset_db = ImageDatabase(root_path=self.root_path)

        img_name_list = list(self.db.keys())
        img_name_list.sort()

        if mode == "random":
            print("- selecting a random subset of size {} with seed {}".format(subset_size, seed))
            np.random.seed(seed)
            subset_indices = np.random.choice(len(img_name_list), subset_size, replace=False)
            subset_indices.sort()
        elif mode == "regular":
            print("- selecting a regular subset of size {}".format(subset_size))
            subset_indices = np.linspace(0, len(img_name_list) - 1, num=subset_size, dtype=int)
        subset_img_names = [img_name_list[i] for i in subset_indices]

        for img_name in subset_img_names:
            subset_db.add_image(img_name, self.db[img_name])
        
        return subset_db


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


def str2list(s):
    """Convert a string to a list of values.

    Parameters:
    s (str): The input string looking as list.

    Returns:
    list: The list of values.

    """

    s = s.translate({ord(c): '' for c in "['']"})
    return [val.strip() for val in s.split(",")]


def merge_databases(db_old, db_new, append_mode):
    """Merge two databases.

    Parameters:
    db_old (ImageDatabase or str): The old database or the path to the old
    database file.
    db_new (ImageDatabase or str): The new database or the path to the new
    database file.
    append_mode (str): The mode of the merging. Can be "keep_new" (keep the 
    new data in case of overlap), "keep_old" (keep the old data in case of 
    overlap), or "overwrite" (overwrite all of the old data).

    Returns:
    ImageDatabase: The merged database.

    """

    if isinstance(db_old, str):
        db_old_path = db_old
        assert os.path.isfile(db_old_path), "The given database file does not exist: " + db_old
        db_old = ImageDatabase()
        db_old.load(db_old_path)
    
    if isinstance(db_new, str):
        db_new_path = db_new
        assert os.path.isfile(db_new_path), "The given database file does not exist: " + db_new
        db_new = ImageDatabase()
        db_new.load(db_new_path)

    if append_mode == "keep_new":
        merged_db = copy.deepcopy(db_new)
        for img_name, img_data in db_old.db.items():
            if img_name not in merged_db:
                merged_db.add_image(img_name, img_data)
    elif append_mode == "keep_old":
        merged_db = copy.deepcopy(db_old)
        for img_name, img_data in db_new.db.items():
            if img_name not in merged_db:
                merged_db.add_image(img_name, img_data)
    elif append_mode == "overwrite":
        merged_db = copy.deepcopy(db_new)
    else:
        assert False, "Invalid append mode: " + append_mode

    return merged_db


def merge_img_list(list_old, list_new, append_mode):
    """Merge two image lists.

    Parameters:
    list_old (list or str): The old image list or the path to the old image
    list file.
    list_new (list or str): The new image list or the path to the new image
    list file.
    append_mode (str): The mode of the file saving if the there are already
    existing files in the given paths.

    Returns:
    list: The merged image list.
    
    """

    if isinstance(list_old, str):
        list_old_path = list_old
        assert os.path.isfile(list_old_path), "The given image list file does not exist: " + list_old
        with open(list_old_path, "rt") as f:
            list_old = f.readlines()
            list_old = [line.strip() for line in list_old]
    
    if isinstance(list_new, str):
        list_new_path = list_new
        assert os.path.isfile(list_new_path), "The given image list file does not exist: " + list_new
        with open(list_new_path, "rt") as f:
            list_new = f.readlines()
            list_new = [line.strip() for line in list_new]

    if append_mode in ["keep_new", "keep_old"]:
        merged_list = list(set(list_old + list_new))
    elif append_mode == "overwrite":
        merged_list = list_new
    else:
        assert False, "Invalid append mode: " + append_mode
    
    return merged_list


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
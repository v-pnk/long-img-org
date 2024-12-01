"""
Check integrity of the images in the database, comparing the MD5 hashes stored 
in the database file to the actual hashes of the image files.

"""


import os
import hashlib
import argparse

from img_db import ImageDatabase


parser = argparse.ArgumentParser(description="Check integrity of the image files.")
parser.add_argument(
    "db_path", 
    type=str, 
    help="The path to the image database file."
)
parser.add_argument(
    "--dataset_root", 
    type=str, 
    help="The root path of the dataset. Directory of the database file by default."
)


def main(args):
    dataset_root = os.path.dirname(args.db_path) if args.dataset_root is None else args.dataset_root
    database = ImageDatabase(root_path=dataset_root)
    database.load(args.db_path)

    for img_relpath in database.db:
        img_path = os.path.join(dataset_root, img_relpath)
        if not os.path.isfile(img_path):
            print("WARN: Image not found: {}".format(img_path))
            continue
        
        md5_db = database.db[img_relpath]["md5"]
        md5_img = hashlib.md5(open(img_path, 'rb').read()).hexdigest()

        if md5_db != md5_img:
            print("WARN: MD5 hash mismatch: {}".format(img_path))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

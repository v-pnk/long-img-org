# Long-term image dataset organization tool
The tool is designed for organizing long-term image datasets and their gradual building with minimal manual effort by extensively using the image metadata.

Available features:

- organizes the given images by date and sensor
- can extract frames from video files
- can use GPX track files
- maintains a database of different image sensors
- identifies image sequences
- can assign multiple tag types to images and sequences
    - day-time (day, night, dawn, dusk)
    - named locations defined by GeoJSON polygons

## Installation
Only a few Python packages are required:

- [PyExifTool](https://pypi.org/project/PyExifTool/)
- [PyYAML](https://pypi.org/project/PyYAML/)
- [NumPy](https://pypi.org/project/numpy/)
- [Shapely](https://pypi.org/project/shapely/)
- [OpenCV](https://pypi.org/project/opencv-python/)

All of them can be installed using pip:

```bash
pip install pyexiftool pyyaml numpy shapely opencv-python
```

PyExifTool requires the [ExifTool](https://exiftool.org/) to be installed on the system. The installation instructions can be found on the [ExifTool website](https://exiftool.org/install.html). ExifTool is also available in the package repositories of most Linux distributions.

```bash
# Ubuntu
sudo apt install exiftool

# Arch Linux
sudo pacman -S perl-image-exiftool

# Fedora
sudo dnf install perl-Image-ExifTool
```

## Usage
### New data processing

The tool works best if run on original images containing full EXIF from a camera or a smartphone. The processing can be done gradually in batches, building the dataset over time.

Basic usage example:

```bash
python3 process.py <path_to_input_image_dir> <path_to_dataset_root>
```

For all the available options, use the `--help` flag:

```bash
python3 process.py --help
```

### Retrieval

You can retrieve subsets of your dataset based on the image metadata stored in the database and save it as a new CSV database, copy the images or create a list of image names / paths:

```bash
python3 img_db.py <path_to_input_image_database.csv> "<list_of_filters>" --db_out <path_to_output_image_database.csv> --img_list <path_to_output_img_list.txt> --img_dir <path_to_output_img_dir>
```

The `<list_of_filters>` is a list of key-value pairs separated by semicolons. Possible keys are:
- `capture_date`
- `capture_year`
- `capture_month`
- `capture_hour`
- `sensor_name`
- `tag_location`
- `tag_daytime`
- `latitude`
- `longitude`
- `altitude`

The values can be defined as a single value (`capture_month: 11`), as a range (`capture_month: [11,2]`), as a maximum / minimum value (`capture_year: 2023]` / `capture_year: [2022`) or as a list (`capture_month: 3,4,5`). A more complex example to retrieve images captured between December and February, during dawn or dusk, and located in the `main_square` location:

```
"capture_month: [12,2]; tag_daytime: dawn,dusk; tag_location: main_square"
```

The script also supports creating random or uniformly sampled subsets of the dataset:

```bash 
python3 img_db.py ... --subset_size <number_of_images_to_samlpe> --subset_mode <random|uniform>
```




## Notes
- The named locations (used for location tagging) are defined by polygons in a GeoJSON file. The map with polygons can be created, e.g., using [uMap](https://umap.openstreetmap.fr/en/).
- The tool creates a specific dataset directory structure:
    - the `<day_directories>` are named by the date in the format `YYYY-MM-DD`
    - the `<sensor_directories>` are named by the sensor name constructed from the EXIF metadata
    - the `metadata.yaml` file contains the definition of sequences and tags for the given day and sensor
    - the `sensors.yaml` file contains the definition of all the sensors used in the dataset
    - the `image_database.csv` file contains information about all the images in the dataset used for quick retrieval

```
<dataset_root>
├── <day_directories>
│   ├── <sensor_directories>
│   │   ├── <image_files>
│   │   ├── <video_files>
│   │   ├── metadata.yaml
├── sensors.yaml
├── image_database.csv
```

- The coordinates of images are extracted from the EXIF metadata (if available) or interpolated using a GPX track file.
- Video files cannot directly contain a location in their metadata, so the locations for the extracted frames are interpolated from a GPX track file and saved in the frame EXIF.
- The tool matches the images to the GPX track file solely by the time of capture. Therefore it cannot handle the case when multiple GPX files corresponding to different sensors capturing at the same time are being processed at once. This case can be solved by processing the corresponding image data and GPX files sequentially (first process the images and GPX files from sensor A, then process the images and GPX files from sensor B, etc.).

## TODO
- [x] image resizing
- [x] global image database
- [x] image retrieval (based on locations, tags, sequences, etc.)
- [x] image integrity checks based on MD5 hash
- [ ] automatic image-based tagging (weather, foliage, etc.)
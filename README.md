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

The tool works best if run on original images containing full EXIF from a camera or a smartphone. The processing can be done gradually in batches, building the dataset over time.

Basic usage example:

```bash
python3 process.py <path_to_input_image_dir> <path_to_dataset_root>
```

For all the available options, use the `--help` flag:

```bash
python3 process.py --help
```

## Notes
- The named locations (used for location tagging) are defined by polygons in a GeoJSON file. The map with polygons can be created, e.g., using [uMap](https://umap.openstreetmap.fr/en/).
- The tool creates a specific dataset directory structure:
    - the `<day_directories>` are named by the date in the format `YYYY-MM-DD`
    - the `<sensor_directories>` are named by the sensor name constructed from the EXIF metadata
    - the `metadata.yaml` file contains the definition of sequences and tags for the given day and sensor
    - the `sensors.yaml` file contains the definition of all the sensors used in the dataset

```
<dataset_root>
├── <day_directories>
│   ├── <sensor_directories>
│   │   ├── <image_files>
│   │   ├── <video_files>
│   │   ├── metadata.yaml
├── sensors.yaml
```

- The locations of images are extracted from the EXIF metadata (if available) or interpolated using a GPX track file.
- Video files cannot directly contain a location in their metadata, so the locations for the extracted frames are interpolated from a GPX track file and saved in the frame EXIF.
- The tool matches the images to the GPX track file solely by the time of capture. Therefore it cannot handle the case when multiple GPX files corresponding to different sensors capturing at the same time are being processed at once. This case can be solved by processing the corresponding image data and GPX files sequentially (first process the images and GPX files from sensor A, then process the images and GPX files from sensor B, etc.).

## TODO
- [x] image resizing
- [x] global image database
- [ ] image retrieval (based on locations, tags, sequences, etc.)
- [ ] automatic image-based tagging (weather, foliage, etc.)
from PIL import Image
from PIL.ExifTags import TAGS

def get_focal_length_pixels(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    
    # Extract values - you fill in the tag names
    focal_length = None
    focal_35mm = None
    image_width = None
    
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == 'FocalLength':
            focal_length = value
        if tag == 'FocalLengthIn35mmFilm':
            focal_35mm = value
        if tag == 'ExifImageWidth':
            image_width = value
    
    # Calculate focal length in pixels
    crop_factor = focal_35mm / focal_length
    sensor_width = 36 / crop_factor
    focal_length_px = (focal_length * image_width) / sensor_width
    
    return focal_length_px

print(get_focal_length_pixels('C:\ML PROJECTS\InfraScan-Sentinel\IMG_4568.JPG'))
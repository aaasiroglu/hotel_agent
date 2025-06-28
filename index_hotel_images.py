import utils
import services
# Your image list:
image_links = [
    f"http://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/{i}.jpg"
    for i in range(1, 26)
]

utils.process_images_to_json_and_insert(image_links)
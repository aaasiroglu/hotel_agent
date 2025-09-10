import utils
import services
# Your image list:
image_links = [
    f"http://......../HotelImages/{i}.jpg"
    for i in range(1, 26)
]

utils.process_images_to_json_and_insert(image_links)

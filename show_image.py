from PIL import Image
import numpy as np

def show_a_image(img ,mode=None):
    img = Image.fromarray(img*255, mode=mode)
    img.show()


def show_images_array(img_array, fpath):
    horizon_units = len(img_array[0])
    vertical_units = len(img_array)
    image_size = img_array[0][0].shape[:2]
    image_width = image_size[0]
    image_height = image_size[1]

    horizon_margin = 2
    vertical_margin = 2

    panel_width = horizon_units*image_width + (horizon_units-1)*horizon_margin
    panel_height = vertical_units*image_height +(vertical_units-1)*vertical_margin

    panel = Image.new('RGB', (panel_width, panel_height), (255, 255, 255))

    horizon_offset = 0
    vertical_offset = 0

    for vertical_unit in img_array:
        for horizon_unit in img_array[vertical_unit]:
            pasted_image = Image.fromarray((horizon_unit*255).astype(np.unit8), mode=None)
            panel.paste(pasted_image, (horizon_offset, vertical_offset))
            horizon_offset += image_width + horizon_margin

        horizon_offset = 0
        vertical_offset += image_height + vertical_margin

    if fpath is not None:
        panel.save(fpath)

    panel.show()


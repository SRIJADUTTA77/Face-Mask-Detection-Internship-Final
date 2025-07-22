import os
import cv2
import xml.etree.ElementTree as ET

ANNOT_DIR = "annotations"
IMAGE_DIR = "images"
CROP_DIR = "cropped_faces"

def extract_and_crop():
    for xml_file in os.listdir(ANNOT_DIR):
        tree = ET.parse(os.path.join(ANNOT_DIR, xml_file))
        root = tree.getroot()

        filename = root.find('filename').text
        image_path = os.path.join(IMAGE_DIR, filename)
        image = cv2.imread(image_path)

        for obj in root.findall('object'):
            label = obj.find('name').text  # 'mask' or 'no_mask'
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Crop and save
            face_crop = image[ymin:ymax, xmin:xmax]
            face_crop = cv2.resize(face_crop, (128, 128))
            save_path = os.path.join(CROP_DIR, label)
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, f"{filename}_{xmin}_{ymin}.png")
            cv2.imwrite(out_file, face_crop)

if __name__ == "__main__":
    extract_and_crop()
label_map = {
    "with_mask": "mask",
    "without_mask": "no_mask",
    "mask_weared_incorrect": "no_mask"  # Optional
}
label = obj.find('name').text
target_label = label_map.get(label)
save_path = os.path.join(CROP_DIR, target_label)
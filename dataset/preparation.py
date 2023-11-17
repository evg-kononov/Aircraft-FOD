import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def drop_duplicate_neighbors(root_path, threshold=0):
    images = []
    images_path = []
    for path, folders, files in os.walk(root_path):
        deleted_num = 0
        for file in tqdm(files):
            if "mp4" not in file:
                file_path = os.path.join(path, file)
                img = Image.open(file_path)
                img = np.asarray(img) / 255
                images.append(img)
                images_path.append(file_path)
            if len(images) > 2:
                del images[0]
                del images_path[0]
            if len(images) == 2:
                if images[0].shape != images[1].shape:
                    continue
                value = images[0] - images[1]
                if abs(value.sum()) <= threshold:
                    os.remove(images_path[0])
                    del images[0]
                    del images_path[0]
                    deleted_num += 1
        print(f"Number of deleted images: {deleted_num}")


def trim(im):
    """
    Этот алгоритм вывел я, удаляем именно черные рамки
    :param im: image to crop
    :return: cropped image
    """
    y1 = 0
    y2 = im.size[1] - 1

    while im.getpixel((0, y1)).count(0) >= 2:
        y1 += 1

    while im.getpixel((0, y2)).count(0) >= 2:
        y2 -= 1

    return im.crop((0, y1, im.size[0], y2))


def apply_trim(root_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    images_names = [i for i in os.listdir(root_path) if i.endswith(".png")]
    for image_name in images_names:
        image = Image.open(os.path.join(root_path, image_name))
        image = trim(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, image_name), image)


if __name__ == "__main__":
    root_path = r"C:\Users\admin\Desktop\DATASET_V2\АСФАЛЬТ\День\1.Мелкий\5.Асфальт\0.normal"
    save_path = r"./0.normal"
    apply_trim(root_path, save_path)
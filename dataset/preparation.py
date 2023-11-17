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


def trim2(im):
    """
    Этот алгоритм вывел я, удаляем именно черные рамки
    :param im: image to crop
    :return: cropped image
    """
    y1 = 0
    y2 = im.size[1] - 1

    while im.getpixel((400, y1)).count(0) >= 2:
        y1 += 1

    while im.getpixel((400, y2)).count(0) >= 2:
        y2 -= 1

    return im.crop((0, y1, im.size[0], y2))


def trim(im):
    """
    Этот алгоритм вывел я, удаляем именно черные рамки
    :param im: image to crop
    :return: cropped image
    """
    y1 = 0
    y2 = im.size[1] - 1

    x1 = 0
    x2 = im.size[0] - 1

    while im.getpixel((400, y1)).count(0) >= 2 or sum(im.getpixel((400, y1))) <= 6:
        y1 += 1

    while im.getpixel((400, y2)).count(0) >= 2 or sum(im.getpixel((400, y2))) <= 6:
        y2 -= 1

    while im.getpixel((x1, 400)).count(0) >= 2 or sum(im.getpixel((x1, 400))) <= 6:
        x1 += 1

    while im.getpixel((x2, 400)).count(0) >= 2 or sum(im.getpixel((x2, 400))) <= 6:
        x2 -= 1

    return im.crop((x1, y1, x2, y2))


def apply_trim2(root_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    images_names = [i for i in os.listdir(root_path) if i.endswith(".png")]
    for image_name in images_names:
        image = Image.open(os.path.join(root_path, image_name))
        image = trim(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, image_name), image)


def apply_trim(root_path):

    for path, folders, files in os.walk(root_path):
        for file in tqdm(files):
            save_path = os.path.join(path, file)
            image = Image.open(save_path)
            image = trim(image)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file, image)
            os.replace(file, save_path)


if __name__ == "__main__":
    root_path = r"C:\Users\admin\PycharmProjects\Aircraft-FOD\dataset\Aircraft-FOD-DS-v3"
    save_path = r"./1.abnormal"
    apply_trim(root_path)
    #img = Image.open(r"C:\Users\admin\PycharmProjects\Aircraft-FOD\dataset\Aircraft-FOD-DS-v3\АСФАЛЬТ\День\1.Мелкий\1.abnormal\day_5_cutted_203.png")
    #print(img.getpixel((300, 0)))

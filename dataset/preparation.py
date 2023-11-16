import os
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


root_path = r"C:\Users\admin\Desktop\DATASET_V2"
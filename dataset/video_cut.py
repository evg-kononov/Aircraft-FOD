import os.path
import cv2


def save_frame_range(video_path, start_frame, stop_frame, step_frame,
                     dir_path, basename, ext="png"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    for n in range(start_frame, stop_frame, step_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
        else:
            return


path = r"C:\Users\admin\Desktop\БЕТОН\Вечер\2.Средний\22.Бетон\22_cutted.mp4"
time = "evening"  # "day" or "evening"
dir = time + "_" + os.path.basename(path)[:-4]

cap = cv2.VideoCapture(path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
save_frame_range(path, 0, frame_count, 1, dir, dir)

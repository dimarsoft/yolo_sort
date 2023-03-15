from ultralytics import YOLO
from sort import *
import argparse
from pathlib import Path
import cv2


def run_video(video_in, model, output_folder):
    video_in = Path(video_in)
    txt_file = Path(output_folder) / f"{video_in.stem}.txt"

    input_video = cv2.VideoCapture(str(video_in))

    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # количество кадров в видео
    frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = Sort()

    # считываем все фреймы из видео
    for frame_id in range(frames_in_video):
        ret, frame_image = input_video.read()

        if frame_id < 61:
            continue
        predict_frame = model.predict(source=frame_image)

        result = predict_frame[0]

        print(f"{frame_id} = {len(predict_frame)}")

        det = result.boxes.cpu().numpy()

        track_bbs_ids = tracker.update(det.boxes)
        print(track_bbs_ids)

    input_video.release()


def run(video_in, model_path, output_folder):
    model = YOLO(model_path)

    video_in = Path(video_in)

    if video_in.is_dir():
        for entry in video_in.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                run_video(entry, model, output_folder)
    else:
        run_video(video_in, model, output_folder)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in', type=Path, help='path to single video or folder')
    parser.add_argument('--model_path', type=Path)
    parser.add_argument('--output_folder', type=Path, help='model.pt path(s)')

    return parser.parse_args()


def main(run_params):
    run(**vars(run_params))


def run_example():
    model = "D:\\AI\\2023\\models\\Yolo8s_batch32_epoch100.pt"
    src_video_path = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"
    output_video_path = "D:\\AI\\2023\\Track\\Sort\\"
    labels_path = "D:\\AI\\2023\\Track"

    run(src_video_path, model, output_video_path)


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    run_example()

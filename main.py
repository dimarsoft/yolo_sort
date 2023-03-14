from ultralytics import YOLO
from sort import *
import argparse
from pathlib import Path


def run_video(video_in, model, output_folder):
    video_in = Path(video_in)
    txt_file = Path(output_folder) / f"{video_in.stem}.txt"

    print(txt_file)
    predict = model.predict(source=video_in)

    tracker = Sort()

    for x in predict:
        track_bbs_ids = tracker.update(x)
        print(track_bbs_ids)


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

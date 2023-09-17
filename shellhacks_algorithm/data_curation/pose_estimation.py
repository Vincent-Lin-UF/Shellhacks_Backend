import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import subprocess
import ffmpeg
import matplotlib.pyplot as plt
import os
import pickle
from .draw_pose import draw_landmarks_on_image


from typing import List, Dict, Tuple, Any


class Detector(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Detector, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path=f'{os.path.dirname(os.path.abspath(__file__))}/../model/pose_landmarker.task')
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(self.options)

    def detect(self, img: mp.Image) -> vision.PoseLandmarkerResult:
        return self.detector.detect(img)



class PoseEstimatePoint:
    def __init__(self, name: str, x: float, y: float):
        self.name = name
        self.pos = (x, y)

    def __repr__(self) -> str:
        return f"{self.name} ({self.pos[0]}, {self.pos[1]})"

    def __str__(self) -> str:
        return repr(self)


POSE_PAIRS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"), # top of torso
    ("LEFT_SHOULDER", "LEFT_ELBOW"), # left upper arm
    ("LEFT_ELBOW", "LEFT_WRIST"), # left forearm
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), # right upper arm
    ("RIGHT_ELBOW", "RIGHT_WRIST"), # right forearm
    ("LEFT_SHOULDER", "LEFT_HIP"), # left torso
    ("LEFT_HIP", "LEFT_KNEE"), # left quad
    ("LEFT_KNEE", "LEFT_ANKLE"), # left calf
    ("RIGHT_SHOULDER", "RIGHT_HIP"), # right torso
    ("RIGHT_HIP", "RIGHT_KNEE"), # right quad
    ("RIGHT_KNEE", "RIGHT_ANKLE")] # right calf


LIMB_PAIRS = [
    (("RIGHT_SHOULDER", "RIGHT_HIP"), ("RIGHT_SHOULDER", "RIGHT_ELBOW")),
    (("LEFT_SHOULDER", "LEFT_HIP"), ("LEFT_SHOULDER", "LEFT_ELBOW")),
    (("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST")),
    (("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST")),
    (("RIGHT_SHOULDER", "RIGHT_HIP"), ("RIGHT_HIP", "RIGHT_KNEE")),
    (("LEFT_SHOULDER", "LEFT_HIP"), ("LEFT_HIP", "LEFT_KNEE")),
    (("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE")),
    (("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"))
]

LIMB_PAIRS_DESCRIPTIONS = [
    "Right arm angle",
    "Left arm angle",
    "Right arm bend angle",
    "Left arm bend angle",
    "Right waist bend angle",
    "Left waist bend angle",
    "Right leg bend angle",
    "Left leg bend angle"
]


"""
Relevant pose locations:
Nose, to represent the center of the head (0) // get rid of
Maybe ears so we can know the orientation of the head (7, 8) // get rid of
Shoulders (11, 12)
Elbows (13, 14)
Hip (23, 24)
Wrists (15, 16)
Knees (25, 26)
Ankles (27, 28)

[
left shoulder
right shoulder
left elbow
right elbow
left wrist
right wrist
left hip
right hip
left knee
right knee
left ankle
right ankle
]
"""

class PoseEstimate:
    def __init__(self):
        self.points = []

    def add_point(self, name: str, x: float, y: float) -> None:
        self.points.append(PoseEstimatePoint(name, x, y))

    def get_point(self, name: str) -> PoseEstimatePoint | None:
        for point in self.points:
            if point.name == name:
                return point

    def serialize(self) -> dict[Any, list[Any]]:
        ret = {}
        for point in self.points:
            ret[point[0]] = [point[1][0], point[1][1]]
        return ret

    def calculate_angle_net(self) -> dict[tuple[tuple[str, str], tuple[str, str]], float | Any] | None:
        if len(self.points) == 0:
            return None

        vecs = {}

        for pair in POSE_PAIRS:
            a = self.get_point(pair[0])
            b = self.get_point(pair[1])

            vecs[pair] = (a.pos[0] - b.pos[0], a.pos[1] - b.pos[1])

        angles = {}

        for pair in LIMB_PAIRS:
            a = vecs[pair[0]]
            b = vecs[pair[1]]

            angle = 180 / np.pi * np.arccos((a[0] * b[0] + a[1] * b[1]) / np.sqrt(a[0] ** 2 + a[1] ** 2) / np.sqrt(b[0] ** 2 + b[1] ** 2))
            angles[pair] = angle

        return angles


class PoseEstimateFactory:
    indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    names = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST',
             'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']

    def __init__(self):
        pass

    @staticmethod
    def create_pose_estimate(img: mp.Image) -> tuple[PoseEstimate, mp.Image | None]:
        detector = Detector()

        raw_pose_estimate = detector.detect(img)

        if len(raw_pose_estimate.pose_landmarks) == 0:
            return PoseEstimate(), None

        res = raw_pose_estimate.pose_landmarks[0]

        annotated_image = draw_landmarks_on_image(img.numpy_view(), raw_pose_estimate)

        if len(res) < 33:
            return PoseEstimate(), None

        ret = PoseEstimate()

        for i, name in zip(PoseEstimateFactory.indices, PoseEstimateFactory.names):
            ret.add_point(name, res[i].x, res[i].y)

        return ret, annotated_image


def create_pose_estimate_from_video(video: List[mp.Image]) -> List[tuple[PoseEstimate, mp.Image]]:
    ret = []

    for frame in video:
        ret.append(PoseEstimateFactory.create_pose_estimate(frame))

    return ret


def create_angle_nets_from_video(video: List[mp.Image]) -> tuple[dict[tuple[tuple[str, str], tuple[str, str]], list[Any]], list[mp.Image]]:
    ret = {}
    ret_annotated = []

    pose_estimates = create_pose_estimate_from_video(video)

    for pose_estimate, annotated_image in pose_estimates:
        if not pose_estimate:
            continue
        if annotated_image is None:
            continue
        an = pose_estimate.calculate_angle_net()

        if not an:
            continue

        ret_annotated.append(annotated_image)

        for key, value in an.items():
            if key not in ret:
                ret[key] = []
            ret[key].append(value)

    return ret, ret_annotated


def process_video(filename: str, compression: int = 1) -> tuple[dict[tuple[tuple[str, str], tuple[str, str]], list[Any]], list[mp.Image]]:
    probe = ffmpeg.probe(filename)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    out, _ = (
        ffmpeg
        .input(filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )

    mp_frames: List[mp.Image] = []

    print(filename)
    print(len(video))

    for frame in video[::compression]:
        mp_frames.append(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))

    return create_angle_nets_from_video(mp_frames)


class VideoProcessor:
    def __init__(self):
        pass

    @staticmethod
    def process_video(filename: str, compression: int = 1) -> tuple[dict[tuple[tuple[str, str], tuple[str, str]], list[Any]], list[mp.Image]]:
        processed = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/../processed_videos/baselines")

        old_filename = filename
        filename = filename.split('/')[-1]

        if filename + ".pickle" in processed:
            with open(f"{os.path.dirname(os.path.abspath(__file__))}/../processed_videos/baselines/{filename}.pickle", "rb") as f:
                obj = pickle.load(f)
                return obj

        res = process_video(old_filename, compression)

        with open(f"{os.path.dirname(os.path.abspath(__file__))}/../processed_videos/baselines/{filename}.pickle", "wb") as f:
            pickle.dump(res, f)

        return res


def main():
    print(VideoProcessor.process_video("../data/bb_trimmed.mp4", compression=5))


if __name__ == "__main__":
    main()

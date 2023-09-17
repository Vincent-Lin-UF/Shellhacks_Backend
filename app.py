import base64
import datetime
import os
import cv2

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from flask_cors import CORS

import ffmpeg

import shellhacks_algorithm


UPLOAD_FOLDER = "static/uploads/"

app = Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3 * 300 * 1024 * 1024


choice_to_video = {
    "Baseball": "shellhacks_algorithm/data/baseball/baseball_side_view.mp4",
    "Basketball": "shellhacks_algorithm/data/basketball/basketball_side_view.mp4",
    "Boxing": "shellhacks_algorithm/data/",
    "Football": "shellhacks_algorithm/data/football/football throw back view.mp4",
    "Golf Side View": "shellhacks_algorithm/data/golf/golf_side_view.mp4",
    "Golf Back View": "shellhacks_algorithm/data/golf/golf_front_view.mp4",
    "Running": "shellhacks_algorithm/data/running/running_side_view.mp4",
    "Push-ups": "shellhacks_algorithm/data/pushups/pushups_trimmedx2_1.mp4",
    "Snatch": "shellhacks_algorithm/data/snatch/snatch_front_view.mp4",
    "Squat": "shellhacks_algorithm/data/squat/squat_side_view.mp4",
    "Clean and Jerk": "shellhacks_algorithm/data/cleanandjerk/clean_and_jerk_front_view.mp4"
}


graph_file_names = [
    "Left arm angle.png",
    "Left arm bend angle.png",
    "Left leg bend angle.png",
    "Left waist bend angle.png",
    "Right arm angle.png",
    "Right arm bend angle.png",
    "Right leg bend angle.png",
    "Right waist bend angle.png"
]


@app.route('/upload', methods=['POST'])
def upload_video():
    r = request.json

    sport = r['currentSport']
    video = r['selectedFile']

    print("sport:", sport)

    video_bytes = base64.b64decode(video.split(',')[1])

    filename = sport + '-' + str(datetime.datetime.now().timestamp())
    filepath = UPLOAD_FOLDER + filename + '.mp4'

    with open(filepath, 'wb') as f:
        f.write(video_bytes)

    scores, annotated_images = shellhacks_algorithm.process_files(choice_to_video[sport], filepath)

    jpegs = []

    for annotated_image in annotated_images:
        jpegs.append(cv2.imencode(".jpg", annotated_image)[1].tobytes())

    process = ffmpeg.input(
        'pipe:', r='10', f='jpeg_pipe', pix_fmt="rgb24"
        ).output('/tmp/video.mp4', vcodec='libx264', pix_fmt="yuv420p").overwrite_output().run_async(pipe_stdin=True)

    for jpeg in jpegs:
        process.stdin.write(jpeg)

    process.stdin.close()
    process.wait()

    with open('/tmp/video.mp4', 'rb') as f:
        b = f.read()
        enc = base64.b64encode(b)

    response = {x + ' score' : y for x, y in scores.items()}

    response['annotated'] = f"data:video/mp4;base64,{str(enc)[2:-1]}"
    response['overall_score'] = sum(scores.values()) / len(scores)

    for graph in graph_file_names:
        with open(graph, "rb") as f:
            cur_graph_bytes = f.read()
            base_64_encoded_image = f'data:image/png;base64,{str(base64.b64encode(cur_graph_bytes))[2:-1]}'
            response[graph.split('.')[0]] = base_64_encoded_image

        os.remove(graph)

    return jsonify(response), 200


@app.route('/admin', methods=["GET"])
def main_page():
    return "HI!"


if __name__ == "__main__":
    app.run()


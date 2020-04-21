import os
import threading

import cv2
import dlib
import imutils
import numpy as np
import pafy as pafy
from flask import Flask, Response
from flask import render_template
from imutils import face_utils

predictor_path = os.environ.get(
    "PREDICTOR_PATH", "/Users/Pavel/Downloads/shape_predictor_68_face_landmarks.dat"
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

youtube_url = os.environ.get("YOUTUBE_URL", "https://youtu.be/Uj56IPJOqWE?t=17")
video = pafy.new(youtube_url)
best = video.getbest()

stream_url = os.environ.get("STREAM_URL")
video_capture = cv2.VideoCapture(stream_url or best.url)

app = Flask(__name__)

outputFrame = None
lock = threading.Lock()


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def detect_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    landmarks = None
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        landmarks = np.take(shape, [30, 8, 36, 45, 48, 54], axis=0)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(
            image,
            "Face #{}".format(i + 1),
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    return landmarks


def head_pose(image, landmarks):
    size = image.shape

    image_points = np.array(landmarks, dtype="double")

    # 3D model points.
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ]
    )

    # Camera internals

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.cv2.SOLVEPNP_ITERATIVE,
    )

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs,
    )

    for p in image_points:
        cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(image, p1, p2, (255, 0, 0), 2)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )


def detect_position():
    global video_capture, outputFrame, lock
    # watch ip camera stream
    while True:
        try:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if frame is None:
                continue
            frame = imutils.resize(frame, width=500)

            lm = detect_landmarks(frame)

            if lm is not None:
                head_pose(frame, lm)

            # Display the resulting frame (optional)
            # cv2.imshow('Video', frame)

            with lock:
                outputFrame = frame.copy()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except KeyError:
            continue
        except KeyboardInterrupt:
            print("program stopped")
            exit(0)

    video_capture.release()
    cv2.destroyAllWindows()


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_position)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=80, debug=True, threaded=True, use_reloader=False)

import cv2

from utils.inference import detect_faces
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
offsets = (30, 60)

# loading models
face_detection = load_detection_model(detection_model_path)

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:

    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        color = (0, 0, 255)

        draw_bounding_box(face_coordinates, rgb_image, color)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

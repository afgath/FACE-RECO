from statistics import mode

import cv2
from PIL import Image
from keras.models import load_model
import numpy as np
import tensorflow as tf

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = '../../trained_models/detection_models/haarcascade_frontalface_default.xml'
gender_model_path = '../../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
gender_labels = get_labels('imdb')
emotion_model_path = '../../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (20, 40)
emotion_offsets = (20, 40)
offsets = (30, 60)

# loading models
face_detection = load_detection_model(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)
emotion_classifier = load_model(emotion_model_path, compile=False)
graph = tf.get_default_graph()

# getting input model shapes for inference
gender_target_size = gender_classifier.input_shape[1:3]
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []


class Makeup_artist(object):
    def __init__(self):
        pass

    def apply_makeup(self, img):

        # Convert RGB to BGR
        bgr_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:
            color = (0, 0, 255)

            draw_bounding_box(face_coordinates, rgb_image, color)

        img = Image.fromarray(rgb_image)

        return img

    def detect_gender(self, img):
        global graph
        with graph.as_default():
            # Convert RGB to BGR
            gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            rgb_image = np.array(img)
            faces = detect_faces(face_detection, gray_image)
            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
                rgb_face = rgb_image[y1:y2, x1:x2]
                try:
                    rgb_face = cv2.resize(rgb_face, (gender_target_size))
                except:
                    continue
                rgb_face = np.expand_dims(rgb_face, 0)
                rgb_face = preprocess_input(rgb_face, False)
                gender_prediction = gender_classifier.predict(rgb_face)
                gender_label_arg = np.argmax(gender_prediction)
                gender_text = gender_labels[gender_label_arg]
                gender_window.append(gender_text)
                if len(gender_window) > frame_window:
                    gender_window.pop(0)
                try:
                    gender_mode = mode(gender_window)
                except:
                    continue
                if gender_text == gender_labels[0]:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, gender_mode,
                          color, 0, -20, 1, 1)
        img = Image.fromarray(rgb_image)
        return img

    def detect_emotion(self, img):
        global graph
        with graph.as_default():
            # Convert RGB to BGR
            bgr_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_image)

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_classifier._make_predict_function()
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

            img = Image.fromarray(rgb_image)

        return img

    def detect_emotion_gender(self, img):
        global graph
        with graph.as_default():
            bgr_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_image)

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
                rgb_face = rgb_image[y1:y2, x1:x2]

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    rgb_face = cv2.resize(rgb_face, (gender_target_size))
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue
                gray_face = preprocess_input(gray_face, False)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                rgb_face = np.expand_dims(rgb_face, 0)
                rgb_face = preprocess_input(rgb_face, False)
                gender_prediction = gender_classifier.predict(rgb_face)
                gender_label_arg = np.argmax(gender_prediction)
                gender_text = gender_labels[gender_label_arg]
                gender_window.append(gender_text)

                if len(gender_window) > frame_window:
                    emotion_window.pop(0)
                    gender_window.pop(0)

                try:
                    emotion_mode = mode(emotion_window)
                    gender_mode = mode(gender_window)
                    print("Emoción: " + emotion_text + " / Género: " + gender_text)
                except:
                    continue

                if gender_text == gender_labels[0]:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, gender_mode,
                          color, 0, -20, 1, 1)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)
            img = Image.fromarray(rgb_image)

        return img

import numpy as np
import logging

from CurveClassifier import CurveClassifier
from ImageManager import ImageManager
from ImageProcessor import ImageProcessor
from ImageClassifier import ImageClassifier
from ImageSegmentor import ImageSegmentor
from config import (
    model_image_classifier_path, image_classifier_size,
    model_image_segmentor_path, image_segmentor_size,
    model_curve_classifier_path, image_curve_classifier_size,
    setup_logger
)

setup_logger()
logging.getLogger('tensorflow').setLevel(logging.ERROR)

images_path = input("Введи директорию с изображениями:\n")
save_directory = input("Введи директорию для сохранения изображений:\n")
image_manager = ImageManager(images_path, save_directory)
image_classifier = ImageClassifier(model_image_classifier_path)
image_segmentor = ImageSegmentor(model_image_segmentor_path)
curve_classifier = CurveClassifier(model_curve_classifier_path)
l = image_manager.load_images()

for img, image_name in l:
    i = ImageProcessor.bgr_to_rgb(img)
    resized = ImageProcessor.resize_images(i, image_classifier_size)
    norm = ImageProcessor.normalize(resized, True)

    input_image = np.expand_dims(norm, axis=0)
    pred_class, conf = image_classifier.predict(input_image)
    logging.info(f"Изображение с именем: {image_name} это {pred_class} с уверенностью {conf}")

    if pred_class == 'graphiks':
        resized = ImageProcessor.resize_images(img, image_segmentor_size)
        norm = ImageProcessor.normalize(resized, False)
        inv = ImageProcessor.invert_image(norm)
        input_image = np.expand_dims(inv, axis=0)
        pred = image_segmentor.predict(input_image)
        logging.info(f"Изображение с именем: {image_name} имеет {len(pred)} кривых")
        for mask in pred:
            resized = ImageProcessor.resize_images(mask, image_curve_classifier_size)
            input_image = np.expand_dims(resized, axis=0)
            pred_class, conf = curve_classifier.predict(input_image)

            image_manager.save_image(img, image_name, pred_class)

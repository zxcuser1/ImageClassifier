import cv2
import logging
from config import setup_logger
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

setup_logger()


class ImageProcessor:

    @staticmethod
    def resize_images(img, size: tuple):
        """
        Изменяет размер изображения в наборе до заданного размера.

        :param img: Исходное изображение
        :param size: Кортеж (ширина, высота) для нового размера.
        :return: Список изображений с измененным размером.
        """
        try:
            resized = cv2.resize(img, size)
            return resized
        except Exception as e:
            logging.error(f"Не удалось изменить размер изображения: {e}")

    @staticmethod
    def convert_to_grayscale(img):
        """
        Преобразует изображение в черно-белые.

        :param img: Исходное изображение
        :return: Черно-белое изображение.
        """

        try:
            grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return grayscale_image
        except Exception as e:
            logging.error(f"Не удалось преобразовать изображение в черно-белое: {e}")

    @staticmethod
    def invert_image(img):
        """
            Инвертирует изображение.

            :param img: Исходное изображение
            :return: Инвертированное изображение.
        """
        try:
            inverted = 1 - img
            return inverted
        except Exception as e:
            logging.error(f"Не удалось инвертировать: {e}")

    @staticmethod
    def bgr_to_rgb(img):
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Не удалось преобразовать BGR в RGB: {e}")

    @staticmethod
    def normalize(img, is_classify_image: bool = False):
        """
            Нормализует изображение.

            :param is_classify_image:
            :param img: Исходное изображение
            :return: Нормализованное изображение.
        """

        try:
            norm = img.astype('float32') / 255.0 if not is_classify_image else preprocess_input(img)
            return norm
        except Exception as e:
            logging.error(f"Не удалось нормализовать: {e}")
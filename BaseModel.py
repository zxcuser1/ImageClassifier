import logging
from abc import ABC, abstractmethod
import tensorflow as tf
from config import setup_logger
import os

setup_logger()


class BaseModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model(model_path)

    def load_model(self, path: str):
        """
        Загружает модель из указанного пути.
        """
        try:
            if not os.path.exists(path):
                logging.error(f"Файл модели не найден: {path}")
                raise FileNotFoundError(f"Файл модели не найден: {path}")

            model = tf.keras.models.load_model(path)
            logging.info(f"Модель успешно загружена из {path}")
            return model
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")
            raise

    @abstractmethod
    def predict(self, image):
        """
        Метод предсказания — должен быть реализован в подклассе.
        """
        pass

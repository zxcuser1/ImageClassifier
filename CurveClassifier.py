import logging

import numpy as np

from config import setup_logger

from BaseModel import BaseModel

setup_logger()


class CurveClassifier(BaseModel):
    _class_names = ['abrupt decrease', 'abrupt increase', 'concave magnification', 'concave reduction', 'consistency',
                    'convex magnification', 'convex reduction', 'linear magnification', 'linear reduction']

    def predict(self, image):
        """
                Делает предсказание для одного изображения.

                :param image: Одно изображение (np.ndarray), уже нормализованное и приведённое к нужному размеру.
                :return: Кортеж: (предсказанный класс, уверенность)
                """
        try:
            predictions = self.model.predict(image)
            predicted_index = np.argmax(predictions[0])
            predicted_class = self._class_names[predicted_index]
            confidence = predictions[0][predicted_index]
            return predicted_class, confidence
        except Exception as e:
            logging.error(f"Ошибка при предсказании: {e}")

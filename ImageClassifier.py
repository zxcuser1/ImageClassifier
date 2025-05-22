import numpy as np
import logging

from BaseModel import BaseModel


class ImageClassifier(BaseModel):
    _class_names = ["chertezh", "graphiks", "other"]

    def predict(self, image) -> tuple[str, float] | None:
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
            return None

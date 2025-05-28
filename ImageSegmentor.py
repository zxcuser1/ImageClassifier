import logging

import cv2
import numpy as np

from BaseModel import BaseModel


def extract_curves(predicted_mask, classes=range(1, 11), min_area=1450):
    """
        Функция по выделению кривых в зависимости от класса
        :param: predicted_mask - предсказанная маска
        :param: classes - количество классов, для данной модели 11 - 10 кривых + оси
        :param: min_area - площадь маски, которая учитывается
        :return: curves - словарь, где для каждого класса выделена маска кривой
    """
    curves = {}
    for cls in classes:
        binary_mask = (predicted_mask == cls).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(binary_mask)

        curves[cls] = []

        for i in range(1, num_labels):
            instance_mask = (labels_im == i).astype(np.uint8)

            if np.sum(instance_mask) > min_area:
                curves[cls].append(instance_mask)

    return curves


class ImageSegmentor(BaseModel):

    def predict(self, image) -> list[np.ndarray]:
        """
               Функция по выделению кривых в зависимости от класса
               :param: image - пред обработанное изображение маска

               :return: result - массив масок кривых с изображения
           """
        try:
            prediction = self.model.predict(image)

            predicted_mask = np.argmax(prediction, axis=-1)[0]

            curve_dict = extract_curves(predicted_mask)
            result = []

            # Снимаем batch и возвращаем в uint8 формат
            image_np = (image[0] * 255).astype(np.uint8)

            for cls, instances in curve_dict.items():
                for idx, instance_mask in enumerate(instances):
                    # Проверим размеры на всякий случай
                    if image_np.shape[:2] != instance_mask.shape:
                        instance_mask = cv2.resize(instance_mask, image_np.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

                    masked = cv2.bitwise_and(image_np, image_np, mask=instance_mask)
                    result.append(masked)

            return result
        except Exception as ex:
            logging.error(f"Ошибка при сегментации изображения {ex}")

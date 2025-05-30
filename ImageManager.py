import os
import logging
import cv2
import numpy

from config import setup_logger

setup_logger()


class ImageManager:
    def __init__(self, load_folder_path: str, save_folder_path: str):
        """
        Инициализирует ImageLoader с путем к папке, содержащей изображения.

        :param load_folder_path: Путь к папке с изображениями.
        :param load_folder_path: Путь к папке для выгрузки результата.
        """
        self.load_folder_path = load_folder_path
        self.save_folder_path = save_folder_path or load_folder_path
        logging.info(f"Инициализация загрузчика изображений с папки: {load_folder_path}")

    def load_images(self) -> list[tuple[numpy.ndarray, str]]:
        """
        Загружает все изображения из указанной папки с использованием OpenCV.

        :return: Список изображений в формате tuple(numpy.ndarray, str).
        """
        images = []

        if not os.path.isdir(self.load_folder_path):
            logging.error(f"Папка по пути {self.load_folder_path} не существует.")
            raise FileNotFoundError(f"Папка по пути {self.load_folder_path} не существует.")

        for file_name in os.listdir(self.load_folder_path):
            file_path = os.path.join(self.load_folder_path, file_name)

            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    logging.debug(f"Загружаем изображение: {file_name}")
                    image = cv2.imread(file_path)

                    if image is None:
                        raise ValueError("Файл не удалось прочитать как изображение.")

                    images.append((image, file_name))
                    logging.info(f"Изображение {file_name} успешно загружено.")
                except Exception as e:
                    logging.error(f"Не удалось загрузить изображение {file_name}: {e}")

        logging.info(f"Загружено {len(images)} изображений.")
        return images

    def save_image(self, image: numpy.ndarray, image_name: str, folder_name: str):
        """
        Сохраняет изображение в папку с классом.
        """
        if not os.path.isdir(self.save_folder_path):
            logging.error(f"Папка по пути {self.save_folder_path} не существует.")
            raise FileNotFoundError(f"Папка по пути {self.load_folder_path} не существует.")

        folder_path = self.save_folder_path + '/' + folder_name
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(folder_path + '/' + image_name, image)
        logging.info(f"Изображение {image_name} сохранено в {folder_path}")

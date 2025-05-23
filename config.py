import logging


def setup_logger():
    """
    Настраивает формат лога
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ],
        force=True
    )


model_image_classifier_path = r'models\VIT.h5'
image_classifier_size = (256, 256)

model_image_segmentor_path = r'models\U-net.h5'
image_segmentor_size = (512, 512)

model_curve_classifier_path = r'models\CNN.h5'
image_curve_classifier_size = (256, 256)

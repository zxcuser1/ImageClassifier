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

import logging


def get_console_logger(name: str = "my_app", level: int = logging.INFO):
    """
    :param name: 日志名称（区分不同模块）
    :param level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
    :return: 配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)  # 设置全局日志级别

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


logger = get_console_logger()
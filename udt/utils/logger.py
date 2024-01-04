import logging


class CustomHandler(logging.StreamHandler):
    def format(self, record) -> str:
        fmt = '{asctime} "{message}" ({pathname}, line {lineno})'
        formatter = logging.Formatter(fmt, datefmt='[%Y.%m.%d %H:%M:%S]', style='{')
        return formatter.format(record)


class Logger:
    def __init__(self, name, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        handler = CustomHandler()
        handler.setLevel(level)
        self.logger.addHandler(handler)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    # for show called line
    @property
    def debug(self):
        return self.logger.debug


def get_logger(name: str, level: int = logging.INFO):
    return Logger(name, level)

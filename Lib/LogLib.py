import os
import logging
from logging.handlers import RotatingFileHandler

class LogManager:
    logger = None

    def __init__(self):
        self.logFilePath = None

    def initialize(self, logFolder, logFileName):
        print(logFolder)
        if os.path.exists(logFolder) == False:
            os.mkdir(logFolder)

        self.logFilePath = os.path.join(logFolder, logFileName)
        #print("[initialize] LogFilePath : ", self.logFilePath)

        # logging 객체 선언
        self.logger = logging.getLogger(self.logFilePath)

        # logger 종류
        # 1) CRITICAL = 50
        # 2) ERROR = 40
        # 3) WARNING = 30
        # 4) INFO = 20
        # 5) DEBUG = 10
        # 6) NOTSET = 0

        # 로그 출력 레벨 설정
        self.logger.setLevel(level = logging.DEBUG)
        # 핸들러 추가 : 파일로 로그 출력 핸들러 선언, 1024 * 10000 = 10MB, log 파일 10개 까지 백업
        fileHandler = RotatingFileHandler(filename = self.logFilePath, maxBytes = 1024 * 10000, backupCount = 10, encoding = "UTF-8")
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] >> %(message)s')
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)
        # 핸들러 추가 : 콘솔 출력 핸들러 선언
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)

        # LogManager 시작시 타이틀 출력
        self.logger.info("\n")
        self.logger.info("------------------------------------------")
        self.logger.info("              Start Logging               ")
        self.logger.info("------------------------------------------")

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(LogManager, cls).__new__(cls)
        return cls.instance

"""
# Test
obj1 = LogManager()
obj1.initialize(logFolder = os.getcwd(),logFileName = "MovementsDXF.log")
obj1.logger.info("1st log obj1")
print(obj1)
print()
obj2 = LogManager()
obj2.logger.info("2st log obj2")
print(obj2)
"""

"""
# 선언 및 사용법
logObj = LogManager()
logObj.initialize(logFolder = os.getcwd(),logFileName = "MovementsDXF.log")
log = logObj.logger
log.debug("msg - debug")
log.info("msg - info")
log.warning("msg - warn")
log.error("msg - error")
log.critical("msg - critical")
#"""






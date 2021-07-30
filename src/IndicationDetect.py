from Lib import LogLib
from Lib import UtilLib
import cv2 as cv
from Lib import OpencvUtilLib as util
import numpy as np
from Common import CV_ConstVar, ConstVar
import glob
import math

class IndicationDetect():
    def __init__(self,img,origin):
        self.img = img
        self.origin = origin

    def Convert2HSV(self):
        self.hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)

    def FirstThreshold(self):
        '''
        * 가장 밝을 경우
        :return: n
        '''
        self.threshold = cv.inRange(self.hsv, (0, 0, 150), (255, 255, 255))
        self.threshold = util.get_threshold(self.threshold, [0, 255], opt=cv.THRESH_OTSU)
        # Visualization
        '''
        util.get_show('1 - threshold Image', self.threshold)
        # '''
        if self.DetectStart():
            return True
        else:
            return False
        
    def SecThreshold(self):
        '''
        * 가장 어두울 경우
        :return: 
        '''
        self.threshold = cv.inRange(self.hsv, (0, 49, 105), (255, 255, 161))
        self.threshold = util.get_threshold(self.threshold, [0, 255], opt=cv.THRESH_OTSU)
        # Visualization
        '''
        util.get_show('2 - threshold Image',self.threshold)
        # '''
        if self.DetectStart():
            return True
        else:
            return False

    def ThirdThreshold(self):
        '''
        * 약간 어두울 경우
        :return:
        '''
        self.threshold = cv.inRange(self.hsv, (0, 0, 183), (112, 255, 255))
        self.threshold = util.get_threshold(self.threshold, [0, 255], opt=cv.THRESH_OTSU)
        # Visualization
        '''
        util.get_show('3 - threshold Image',self.threshold)
        # '''
        if self.DetectStart():
            return True
        else:
            return False

    def FourThreshold(self):
        '''
        * 약간 어두울 경우
        :return:
        '''
        self.threshold = cv.inRange(self.hsv, (0, 26, 70), (15, 255, 255))
        self.threshold = util.get_threshold(self.threshold, [0, 255], opt=cv.THRESH_OTSU)
        # Visualization
        '''
        util.get_show('4 - threshold Image',self.threshold)
        # '''
        if self.DetectStart():
            return True
        else:
            return False

    def FiveThreshold(self):
        '''
        * 약간 어두울 경우
        :return:
        '''
        self.threshold = cv.inRange(self.hsv, (166, 0, 215), (179, 255, 255))
        self.threshold = util.get_threshold(self.threshold, [0, 255], opt=cv.THRESH_OTSU)
        # Visualization
        '''
        util.get_show('5 - threshold Image',self.threshold)
        # '''
        if self.DetectStart():
            return True
        else:
            return False

    def SixThreshold(self):
        '''
        * 약간 어두울 경우
        :return:
        '''
        self.threshold = cv.inRange(self.hsv, (166, 0, 0), (179, 255, 255))
        self.threshold = util.get_threshold(self.threshold, [0, 255], opt=cv.THRESH_OTSU)
        # Visualization
        # '''
        # util.get_show('6 - threshold Image',self.threshold)
        # '''
        if self.DetectStart():
            return True
        else:
            return False

    def SevenThreshold(self):
        '''
        * 약간 어두울 경우
        :return:
        '''
        self.threshold = cv.inRange(self.hsv, (112, 0, 119), (179, 17, 255))
        self.threshold = util.get_threshold(self.threshold, [0, 255], opt=cv.THRESH_OTSU)
        # Visualization
        # '''
        util.get_show('7 - threshold Image',self.threshold)
        # '''
        if self.DetectStart():
            return True
        else:
            return False

    def EightThreshold(self):
        '''
        * 약간 어두울 경우
        :return:
        '''
        self.threshold = cv.inRange(self.hsv, (125, 0, 126), (179, 255, 255))
        self.threshold = util.get_threshold(self.threshold, [0, 255], opt=cv.THRESH_OTSU)
        # Visualization
        # '''
        util.get_show('7 - threshold Image',self.threshold)
        # '''
        if self.DetectStart():
            return True
        else:
            return False

    def DetectStart(self):
        ptsList = list()
        self.boxList = list()
        contour = util.get_contour_line(self.threshold, opt1=cv.RETR_EXTERNAL, opt2=cv.CHAIN_APPROX_NONE)

        for cnt in contour:
            # contour 라인 그리기
            approx = cv.approxPolyDP(cnt, cv.arcLength(cnt, True) * 0.01, True)

            # 꼭지점
            vtc = len(approx)

            # 꼭지점이 4개인 경우만 1순위로 검출
            if vtc == 4:
                # 표시목의 크기 조건
                DetectAreaSize = self.DetectAreaSize(cnt)

                if DetectAreaSize:
                    # util.get_countour_draw(cnt, self.img)
                    rect = cv.minAreaRect(cnt)
                    box = cv.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
                    box = np.int0(box)
                    cv.drawContours(self.img, [box], 0, (0, 0, 255), 2)
                    for i in [box]:
                        for t in range(len(box)):
                            self.boxList.append(tuple(i[t]))
                    print(f'============================표시목 발견==============================')
                    # 표시목이 검출되었을 경우
                    for j in cnt:
                        # 사각형으로 판별된 꼭지점들만 리스트로 담기
                        ptsList.append(tuple(j[0]))

        if not ptsList:
            # 리스트 None 체크
            return False
        else:
            # Visualization
            # '''
            cv.circle(self.origin, max(self.boxList), 1, (0, 0, 255), 3)
            cv.circle(self.origin, min(self.boxList), 1, (0, 0, 255), 3)

            # 왼쪽 시작점
            x1, y1 = min(self.boxList)
            # 오른쪽 끝점
            x2, y2 = max(self.boxList)

            # 대각선 길이 루트(a제곱 + b제곱)
            self.standardImage_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # util.get_show('Sample',self.img)
            # '''

            # print(f'표시목 길이 - {int(self.standardImage_length)}')
            return True


    def CheckThresHoldLevel(self):
        # 가장 밝은 상황
        # 일상적인 상황
        # 역광
        if type(self.img) is not type(None):
            self.Convert2HSV()
            if self.FirstThreshold():
                print(f'1단계 검출 성공')
                return int(self.standardImage_length)
            elif self.SecThreshold():
                print(f'2단계 검출 성공')
                return int(self.standardImage_length)
            elif self.ThirdThreshold():
                print(f'3단계 검출 성공')
                return int(self.standardImage_length)
            elif self.FourThreshold():
                print(f'4단계 검출 성공')
                return int(self.standardImage_length)
            elif self.FiveThreshold():
                print(f'5단계 검출 성공')
                return int(self.standardImage_length)
            elif self.SixThreshold():
                print(f'6단계 검출 성공')
                return int(self.standardImage_length)
            elif self.SevenThreshold():
                print(f'7단계 검출 성공')
                return int(self.standardImage_length)
            elif self.EightThreshold():
                print(f'8단계 검출 성공')
                return int(self.standardImage_length)

    def DetectAreaSize(self,pts):
        '''
        * 표시목만 검출
        :param img: 원본이미지
        :param pts: 검출된 사각형들의 포인트
        :return: 테두리가 그려진 이미지
        '''
        # FIXME: 200 pixel 차이 조정
        MIN = 2600
        MIN_SMALL = 2400
        MAX = 7100
        AreaWidth = 85
        if pts is not None:
            (x,y,w,h) = cv.boundingRect(pts)
            area = w * h
            print(area,w,h)
            if  MIN < area < MAX and w < AreaWidth and h < AreaWidth:
                # Visualization
                '''
                cv.circle(self.img, pt1, 1, (0, 0, 255), 3)
                cv.circle(self.img, pt2, 1, (0, 0, 255), 3)
                cv.rectangle(self.img, pt1, pt2, (0,255,0),2)
                # '''
                return True
            elif MIN_SMALL < area < MAX and w < AreaWidth and h < AreaWidth:
                # Visualization
                '''
                cv.circle(self.img, pt1, 1, (0, 0, 255), 3)
                cv.circle(self.img, pt2, 1, (0, 0, 255), 3)
                cv.rectangle(self.img, pt1, pt2, (0,255,0),2)
                # '''
                return True
            else:
                return False



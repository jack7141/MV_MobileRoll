from Lib import LogLib
import cv2 as cv
from Lib import OpencvUtilLib as util
import numpy as np
import src.FindRoll as fr
import math
from src.IndicationDetect import IndicationDetect as Indication


class RollDetection():
    def __init__(self):
        self.img = None
        self.ImagePath = None
        self.origin = None

    def LoadImageNoneCheck(self):
        # List None Check
        if not self.ImageFileList:
            return False
        else:
            return True

    def ResizeImage(self):
        self.img = util.get_resize(self.img, Ratio=[0.5,0.5])
        self.origin = self.img

    def Image2Gray(self):
        '''
        * 이미지 gray scale로 변환
        '''
        self.gray = util.get_gray(self.img)

    def Image2Median(self):
        '''
        * 이미지 medianBlur처리
        '''
        self.median = cv.medianBlur(self.gray, 5)

    def Image2Threshold(self):
        '''
        * 이미지 threshold로 이진화
        '''
        self.threshold = util.get_threshold(self.median, thres=[80, 255])
        # util.get_show('canny',util.get_canny(self.threshold,thres=[0,200]))

    def ImageErode(self):
        '''
        * 이미지 팽창
        '''
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
        self.threshold = cv.erode(cv.dilate(self.threshold, kernel, iterations=1), kernel, iterations=1)

    def DrawContour(self):
        '''
        * 이미지 외곽선 추출
        '''
        self.contours = util.get_contour_line(self.threshold, opt1=cv.RETR_LIST, opt2=cv.CHAIN_APPROX_NONE)
        cv.drawContours(self.img, self.contours, -1, (255, 0, 0), 2)

    def BiggestCircleDetect(self,circles):
        '''
        * 찾아낸 원들 중에서 가장 반지름의 길이가 긴것을 선택한다
        :param circles: 원의 정보
        :return: x,y좌표, 반지름
        '''
        x_axis = None
        y_axis = None
        radius = 0
        for i in circles[0, :]:
            x_axis, y_axis, r = i[0], i[1], i[2]
            if 400 < y_axis < 800 and 400 < x_axis < 800:
                if radius < r:
                    radius = r
                    x_axis = i[0]
                    y_axis = i[1]

        return x_axis, y_axis, radius


    def GrabCutCircle(self):
        gray=util.get_gray(self.img)
        mask=util.get_threshold(gray,thres=[100,255])
        mask_inv = cv.bitwise_not(mask)
        self.img=cv.bitwise_and(self.img,self.img,mask=mask_inv)

    def DetectAreaSize(self,img,pts):
        '''
        * 표시목만 검출
        :param img: 원본이미지
        :param pts: 검출된 사각형들의 포인트
        :return: 테두리가 그려진 이미지
        '''

        MIN = 3700
        MAX = 7000
        AreaWidth = 80
        if pts is not None:

            (x,y,w,h) = cv.boundingRect(pts)

            # 오른쪽 시작점
            pt1 = (x,y)

            # 오른쪽
            pt2 = (x+w, y+h)
            area = w * h
            if  MIN < area < MAX and w < AreaWidth and h < AreaWidth:
                # cv.rectangle(img,pt1, pt2, (0,255,0),2)
                # print(f'표시목 너비:{w * h}')
                return True
            else:
                return False


    def PixelToCM(self,pts1, pts2):
        '''
        * (지름[pixel] * 7.8[cm])/포스트잇 너비[pixel]
        ==> (실제)포스트잇 대각선 길이 = 10.5
        :param pts1: 좌측 꼭지점 좌표, pts2: 우측하단 꼭지점 좌표
        :return:
        '''
        if pts1 is not None:
            x1, y1 = pts1
            x2, y2 = pts2

            # 대각선 길이 루트(a제곱 + b제곱)
            self.standardImage_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            # 심지 - 10
            # print(f'심지 길이 : {self.CircleDiameter * 10 / self.standardImage_length} CM')
            # print(f'원의 지름 - {self.CircleDiameter}')
            # print(f'원의 반지름 - {self.CircleDiameter/2}')

    def detectRectangle(self):
        '''
        * 표시목 검출
        :param img:
        :return:
        '''
        ptsList = list()

        if self.img is not None:
            # 감도 조절
            # self.img = self.img + 20
            hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
            threshold = cv.inRange(
                hsv, (0, 0, 130), (255, 255, 255))
            threshold = util.get_threshold(threshold, [0, 255], opt=cv.THRESH_OTSU)
            contour = util.get_contour_line(threshold, opt1=cv.RETR_EXTERNAL, opt2=cv.CHAIN_APPROX_NONE)

            for cnt in contour:
                # contour 라인 그리기
                approx = cv.approxPolyDP(cnt, cv.arcLength(cnt, True) * 0.02, True)

                # 꼭지점
                vtc = len(approx)

                # 꼭지점이 4개인 경우만 1순위로 검출
                if vtc == 4:

                    # 표시목의 크기 조건
                    DetectAreaSize = self.DetectAreaSize(self.img, cnt)

                    if DetectAreaSize:

                        print(f'============================표시목 발견==============================')
                        # 표시목이 검출되었을 경우

                        for j in cnt:
                            # 사각형으로 판별된 꼭지점들만 리스트로 담기
                            ptsList.append(tuple(j[0]))

                        # 제일 끝점간의 꼭지점 표시
                        cv.circle(self.img, max(ptsList), 1, (0, 0, 255), 3)
                        cv.circle(self.img, min(ptsList), 1, (0, 0, 255), 3)
                        # 심지 계산
                        self.PixelToCM(min(ptsList), max(ptsList))
                else:
                    # 초기화
                    ptsList.clear()

    def DrawCircle(self):
        '''
        * 홍채 인식 알고리즘 Strat
        '''
        self.RollCenter, self.Rollradius = fr.find_roll(self.gray, daugman_start=400, daugman_end=500, daugman_step=10, points_step=20)


    def HoughCircle(self):
        gray = util.get_gray(self.img)
        # gray = cv.medianBlur(gray, 5)
        gray = util.get_dilation(gray,kernel=[5,5])
        # util.get_show('gray',gray)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 2, rows/8,
                                   param1=50, param2=100, minRadius=50, maxRadius=500)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x_axis, y_axis, r = i[0], i[1], i[2]
            if 500 < y_axis < 600 and 500 < x_axis < 600:
                # 점의 중심선정
                self.CircleDiameter = r * 2
                # print(f'원의 지름 PIXEL - {r*2}')
                cv.circle(self.origin, (x_axis, y_axis), r, (0, 255, 0), 2)
                cv.circle(self.origin, (x_axis, y_axis), 1, (0, 255, 255), 2)

    def roiMasking(self):

        x, y = self.RollCenter

        height, width, channel = self.img.shape
        # 직경계산
        # self.CircleDiameter = self.Rollradius * 2

        # 배경색 지정
        mask = np.zeros(self.img.shape, np.uint8)

        # 선분 두께
        border = 6

        # 원 중심
        # cv.circle(self.img, self.RollCenter, 1, (0, 100, 100), 3)

        # 테두리 색상
        # cv.circle(mask, (x, y), self.Rollradius, (0, 0, 255), border)

        # 원 내부 색상
        cv.circle(mask, (x, y), self.Rollradius, (255, 255, 255), -1)

        # 검은 배경으로 처리
        self.img = cv.bitwise_and(self.img, mask)

        try:
            x1 = max(x - self.Rollradius - border // 2, 0)
            x2 = min(x + self.Rollradius + border // 2, width)
            y1 = max(y - self.Rollradius - border // 2, 0)
            y2 = min(y + self.Rollradius + border // 2, height)

            # 최초에 이미지 내에서 가장 큰 원만 추출
            self.ROI = self.img[y1:y2, x1:x2]

        except:
            print(f'원이 1사분면을 넘어감')
            pass


    def detectCircle(self):
        self.Image2Gray() # image gray scale로 변화
        self.Image2Median() # image gray scale로 변화
        self.Image2Threshold() # 이진화
        self.ImageErode() # image 이진화 팽창
        self.DrawCircle() # 홍채인식 알고리즘
        self.roiMasking() # image gray scale로 변화
        # self.detectRectangle()
        IndicationObj = Indication(self.img,self.origin)
        self.standardImage_length = IndicationObj.CheckThresHoldLevel()
        self.GrabCutCircle()
        self.HoughCircle()
        try:
            print(f'심지 길이 : {int(self.CircleDiameter * 10.5 / self.standardImage_length)} CM')
            print(f'표시목 길이 - {self.standardImage_length}')
            print(f'원의 지름 - {self.CircleDiameter}')
            print(f'원의 반지름 - {self.CircleDiameter / 2}')
            print(f'==================================================================')
            # return 심지길이
        except:
            print(f'============= 심지길이를 측정할 수 없습니다!!!!!! =============')
            print(f'원의 지름 - {self.CircleDiameter}')
            print(f'원의 반지름 - {self.CircleDiameter / 2}')
            # return None
            pass


    def ImageLoop(self):
        self.img = cv.imread(self.ImageFileList[0])
        print(f'File Name - {self.ImageFileList[0]}')
        self.ResizeImage()
        self.detectCircle()
        # 전처리로 이미지 내부에서 심지가 아닌 부분은 삭제하고 처리 시작해야함
        cv.imshow('PRESS P for Previous, N for Next Image', self.origin)

        cv.namedWindow('PRESS P for Previous, N for Next Image')

        i = 0

        while (1):
            k = cv.waitKey(1) & 0xFF
            if k == ord('n'):
                # 다음 사진 Show
                i += 1

                self.img = cv.imread(self.ImageFileList[i % len(self.ImageFileList)])
                print(f'File Name - {self.ImageFileList[i % len(self.ImageFileList)]}')
                self.ResizeImage()
                # 원찾기 시작
                self.detectCircle()
                # util.get_ImageSave(self.ImageFileList[i % len(self.ImageFileList)],self.img)
                cv.imshow('PRESS P for Previous, N for Next Image', self.origin)

            elif k == ord('p'):
                # 이전 사진 Show
                i -= 1

                self.img = cv.imread(self.ImageFileList[i % len(self.ImageFileList)])
                self.ResizeImage()
                # 원찾기 시작
                self.detectCircle()
                cv.imshow('PRESS P for Previous, N for Next Image', self.origin)

            elif k == 27:
                cv.destroyAllWindows()
                break

    # Desktop Version
    def StartDetect(self):
        if (self.LoadImageNoneCheck()):
            self.ImageLoop()
        else:
            print('Image file "' + self.ImagePath + '" could not be loaded.')


    # Android Version
    def running(self,img):
        '''
        * 안드로이드 Bridge로부터 받아온 image의 유무를 파악해서 알고리즘 시작
        parma : image
        return : 심지결과, image
        '''
        if type(img) is not type(None):
            print(f'ImageProcessing Start')
            self.img = img
            self.ResizeImage() # image resize
            self.detectCircle() # 배수재 찾기 Start
            try:
                outSideDiameter = int(self.CircleDiameter * 10.5 / self.standardImage_length)
                inSideDiameter = 14
                thickness = 0.4
                result = ((outSideDiameter**2 - inSideDiameter**2) * 3.14)/(4*thickness)
                return int(result/100), self.origin
            except:
                print('Calculate Error')
                return None
        else:
            print(f'Cant Find Image Path - {img}')
            return None

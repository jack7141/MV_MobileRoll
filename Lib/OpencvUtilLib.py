import cv2 as cv
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import shutil
'''
* opencv 목차
#---------------------------모든 함수 이름은 스네이크 표기법을 사용했음

# x좌표
# y좌표
# 크기조절(resize) 비율 
# 사용자 지정 비율조정
# gray 변환
# 임계치조정
# 팽창
# 침식
# canny filer
# 가로선 강조





*******************************필요한 항목******************************

# FIXME:라인 그려주는 함수 생성

'''


def get_x(img):
    # param - 이미지
    # 이미지의 x값의 총 길이
    # return: 이미지의 x의 총 길이
    return img.shape[1]


def get_y(img):
    # param - 이미지
    # 이미지의 y값의 총 길이
    # return: 이미지의 y의 총 길이
    return img.shape[0]


def get_resize(img, Ratio=[0.3, 0.3]):
    # param - 이미지, Ration(조정 비율)
    # 이미지를 비율에 맞춰서 조정한다
    # fx - 원본 이미지 너비의 0.3
    # fy - 원본 이미지 너비의 0.3
    # interpolatrion - INTER_AREA(쌍선형 보간법)
    # return: 원본 이미지로부터 줄어든 이미지 반환
    try:
        img = cv.resize(img, dsize=(0, 0),  fx=Ratio[0], fy=Ratio[1],
                        interpolation=cv.INTER_AREA)
        return img
    except:
        pass


def get_ImageSave(path, img):
    '''
    * 이미지를 저장
    * param : 경로, 이미지
    '''
    try:
        cv.imwrite(path, img)
    except:
        print(f'[OpenCVUtil] - Image Save Error')
        pass


def get_desize(img, dsize=[0, 0]):
    '''
    * param - 이미지, 사용자 지정 사이즈(dsize)
    # 이미지를 사용자가 지정한 사이즈로 지정해서 resizing한다
    # dsize[0] - x
    # dsize[1] - y
    # interpolatrion - INTER_AREA(쌍선형 보간법)
    # return: 원본 이미지로부터 줄어든 이미지 반환
    '''
    try:
        img = cv.resize(img, dsize=(
            dsize[0], dsize[1]), interpolation=cv.INTER_AREA)
    except:
        pass
    return img


def get_gray(img):
    # param - 이미지
    # return: 원본 이미지로부터 줄어든 이미지 반환
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def get_circle(img):
    '''
    * 예제 - https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html
    :param img: 
    * img : image(이진화 데이터, 8bit포함==> gray)
    * 옵션
    * dp : 이미지 해상도 숫자가 클수록 작은 크기를 가진 해상도를 가진 결과를 산출
    * minDist: 검출한 원의 가운데와 가장 가까운거리, 작으면 하나의 원에서 여러 이웃한 작은 원들로 산출된다
    * param1 : CV_HOUGH_GRADIENT의 경우 높은 임계값을 주는 에지 검출 방법인 canny로 산출한다.
    * parma2 : 이것이 작으면 실패율이 높아진다. 자잘한것까지 모두 찾음
    * minRadius : 최소 반지름
    * maxRadius : 최대 반지름
    :return: 원 데이터
    '''
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, rows / 8,
                              param1=200, param2=200,
                              minRadius=300, maxRadius=500)
    return circles

def get_threshold(img, thres=[0, 0], opt=cv.THRESH_BINARY):
    '''
    * param - 이미지, 경계값(0~255), opt = 옵션

    # 옵션종류
    # THRESH_BINARY
    # THRESH_BINARY_INV
    # THRESH_TRUNC
    # THRESH_TOZERO
    # THRESH_TOZERO_INV

    # return: 원본 이미지로부터 임계치가 적용된 이미지
    '''
    _, thr = cv.threshold(img, thres[0], thres[1], opt)
    return thr


def get_countour_draw(cnt,img):
    '''
    * contour검출 함수를 통해 나온 값들을 이용해서 테두리를 그려준다
    :param cnt: findContours를 통해서 나온 값들
    :param img: 이미지
    :return: 이미지
    '''
    for j in cnt:
        cv.circle(img, tuple(j[0]), 1, (0, 0, 255), 1)
    return img



def get_dilation(img, kernel=[3, 3], iter=1):
    # 팽창 ==> 배경을 팽창시켜서 전경을 줄어들이게 한다
    # pramas : 이미지, 커널사이즈, 횟수
    # return : 변화된 이미지 사이즈
    kernel = np.ones((kernel[0], kernel[1]), np.uint8)
    img = cv.dilate(img, kernel, iterations=iter)
    return img


def get_erosion(img, kernel=[3, 3], iter=1):
    # 침식 ==> 배경을 침식시켜서 전경을 늘리게 한다
    # pramas : 이미지, 커널사이즈, 횟수
    # return : 변화된 이미지 사이즈
    kernel = np.ones((kernel[0], kernel[1]), np.uint8)
    img = cv.erode(img, kernel, iterations=iter)
    return img


def get_canny(img, thres=[50, 150], apertureSize=3):
    # pramas : 이미지, 임계값, 픽셀사이즈
    # return : canny filer 적용된 이미지
    img = cv.Canny(img, thres[0], thres[1], apertureSize=3)
    return img

def get_namewindow(title):
    cv.namedWindow(title)

def get_show(title, img):
    # param - 이미지 윈도우 타이틀, 이미지
    try:

        cv.imshow(title, img)
        cv.waitKey()
        cv.destroyAllWindows()

    except:
        pass

def get_read(filepath):
    return cv.imread(filepath)


def get_pltshow(img):
    plt.imshow(img)
    plt.show()


def ImageLoop(filelist):

    img = cv.imread(filelist[0])
    cv.imshow('PRESS P for Previous, N for Next Image', img)
    cv.namedWindow('PRESS P for Previous, N for Next Image')
    i = 0

    while (True):
        k = cv.waitKey(1) & 0xFF
        if k == ord('n'):
            # 다음 사진 Show
            i += 1
            img = cv.imread(filelist[i % len(filelist)])
            # * 이미지 리사이징할시 위치
            cv.imshow('PRESS P for Previous, N for Next Image', img)
        elif k == ord('p'):
            # 이전 사진 Show
            i -= 1
            img = cv.imread(filelist[i % len(filelist)])
            # * 이미지 리사이징할시 위치
            cv.imshow('PRESS P for Previous, N for Next Image', img)

        elif k == 27:
            cv.destroyAllWindows()
            break

def get_horizantal_line(img):
    '''
    *params: gray 이미지
    return: 가로선만 강조된 이미지
    '''
    img = cv.bitwise_not(img)
    bw = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                              cv.THRESH_BINARY, 15, -2)
    horizontal = np.copy(bw)

    cols = horizontal.shape[1]

    horizontal_size = cols // 30

    # 가로선만 확대
    horizontalStructure = cv.getStructuringElement(
        cv.MORPH_RECT, (horizontal_size, 1))
    # 가로선 팽창 침식 알고리즘 적용
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)

    return horizontal


def get_vertical_line(img):
    '''
    *params: gray 이미지
    return: 가로선만 강조된 이미지
    '''
    img = cv.bitwise_not(img)
    bw = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                              cv.THRESH_BINARY, 15, -2)
    vertical = np.copy(bw)

    rows = vertical.shape[0]

    vertical_size = rows // 30

    # 가로선만 확대
    verticalStructure = cv.getStructuringElement(
        cv.MORPH_RECT, (1, vertical_size))
    # 가로선 팽창 침식 알고리즘 적용
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)

    return vertical


def get_contour_line(img, opt1=cv.RETR_EXTERNAL, opt2=cv.CHAIN_APPROX_SIMPLE):
    '''
    컨투어 생성
    * pramas : binary image
    return contour
    '''
    contours, hierarchy = cv.findContours(img, opt1, opt2)

    return contours


def get_circle(img, pos=[500, 500], size=5):
    '''
    점그리기
    * pramas : image, x,y위치, 점의 사이즈
    return img
    '''
    img = cv.circle(img, (pos[0], pos[1]), size, (0, 0, 255), -1)
    return img


def get_cut_roi(img, x, y, w, h):
    '''
    *ROI 이미지 
    * pramas : image, x,y, 너비, 높이
    return roi
    '''
    roi = img[y:y+h, x:x+w]
    return roi


'''
* os 관련 목차

# 경로 만들기
# 입력한 디렉토리의 모든 항목의 경로 
# 폴더경로 가져오기
# 파일경로, .ext 가져오기
# 파일, 폴더 유무확인
# 폴더 생성하기
'''


def get_new_path(path, add):
    """
    * 새롭게 경로를 만듬
    :param path: 기본경로
    :param add: 파일명 혹은 뒷부분 경로 추가
    :return: 새롭게 만들어진 경로
    """
    return os.path.join(path, add)


def get_all_elements_path_list(path):
    """
     * 입력한 디렉토리의 모든 항목(파일+디렉토리)의 경로를 리스트로 반환
    :param path: 조회할 디렉토리 전체 경로
    :return: 모든 항목들의 경로 리스트
    """
    result = list()
    fileNames = os.listdir(path)
    for fileName in fileNames:
        fullPath = os.path.join(path, fileName)
        result.append(fullPath)
    return result


def get_path(path):
    # param - 폴더의 경로
    # return: c:\ -> c:/
    path = path.replace(os.sep, '/')
    return path


def get_file_ext_name(path):
    '''
    # param - 경로
    # return: 파일이름, .ext
    '''
    f = open(path)
    file_name, ext = os.path.splitext(path)
    file_name = os.path.basename(f.name)
    return file_name.split(".")[0], ext


def get_dir_path(path):
    '''
    # param - 경로
    # return: 폴더경로
    '''
    return os.path.dirname(path)


def get_file_extension(filePath):
    """
    * 파일 확장자 추출
    :param filePath: 파일 전체 경로
    :return: 파일의 확장자
    """
    return os.path.basename(filePath.split(".")[-1])


def get_file_name(filePath):
    """
    * 확장자 포함 파일명 추출
    :param filePath: 파일 전체 경로
    :return:  확장자 포함 파일명
    """
    return os.path.basename(filePath)


def get_exists(file):
    '''
    * 파일, 폴더의 존재유무 확인
    params: 파일의 절대 경로 및 폴더의 절대 경로 
    return: True, False
    '''
    return os.path.exists(file)


def create_folder(directory):
    '''
    *폴더 생성
    params: 디렉토리 경로
    '''

    try:

        if not os.path.exists(directory):

            os.makedirs(directory)

    except OSError:

        print("Error OpencvUtilLib.create_folder : Creating directory" + directory)


def get_new_path(path, add):
    """
    * 새롭게 경로를 만듬
    :param path: 기본경로
    :param add: 파일명 혹은 뒷부분 경로 추가
    :return: 새롭게 만들어진 경로
    """
    return os.path.join(path, add)


def get_combine_list(ListData):
    '''
    * param: 리스트내부 2중 리스트
    * return: 하나의 리스트
    '''
    result = list()
    for elements in ListData:
        for element in elements:
            result.append(element)
    print(result)
    return result


def get_coordinate_list(ListData, axis='x'):
    '''
    * param: 리스트내부 2중 리스트, x 좌표
    * return: 하나의 리스트
    '''
    result = list()
    print(ListData)
    try:
        for i in ListData:
            if axis == 'x':
                # x 좌표
                result.append(i[0])
            elif axis == 'y':
                # y 좌표
                result.append(i[1])

        return result
    except:
        pass


'''
*---------------------------수학 함수 목차
# 평균화
'''


def get_average(value):
    # parmas: value
    # return: 값 평균화
    return np.average(value, axis=0)


'''
*---------------------------os 목차

'''


def get_copy(src, dst):
    shutil.copy(src, dst)

def get_remove_all_files(path,package="*.jpg"):
    # 모든 파일 지정된 경로안의 모든 파일 삭제
    '''
    :param path: 파일이 저장된 위치
    EX) C://Users//movements-image//MV_ROLL//ROLL_IMG//
    :param package: 확장자명
    EX) *.jpg, *.jpeg, *.png... ETC
    :return: None
    '''
    if os.path.exists(path):
        [os.remove(f) for f in glob.glob(f"{path}{package}")]


def get_file_list(path,package="*.jpg"):
    '''
    * 지정한 경로 내부의 모든 파일을 가지고 온다(절대경로로 경로/파일이름 이런식으로 가지고옴)
    :param path: 지정한 경로
    :param package: 기본으론 jpg를 사용하고있음
    :return:
    EX) ['C:\\Users\\movements-image\\MV_ROLL\\ROLL_IMG\\KakaoTalk_20210331_123008162.jpg', 'C:\\Users\\movements-image\\MV_ROLL\\ROLL_IMG\\KakaoTalk_20210331_123008850.jpg']
    '''
    if os.path.exists(path):
        result = [f for f in glob.glob(f"{path}{package}")]

    return result

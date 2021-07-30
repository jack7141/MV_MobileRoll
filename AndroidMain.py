import base64
import cv2 as cv
import numpy as np
from PIL import Image
import io
from src.RollDetection import RollDetection as FullVersion

def AndroidPythonBridge(data):
    '''
    * 안드로이드 BRIDGE
    parma : byte 데이터
    return : 심지측정결과, image
    '''
    # FiXME: ANDROID TEST
    decoded_data = base64.b64decode(data)
    np_data = np.fromstring(decoded_data, np.uint8)

    img = cv.imdecode(np_data, cv.IMREAD_UNCHANGED)
    FullVersionObj = FullVersion()
    result, image = FullVersionObj.running(img)
    pil_im = Image.fromarray(image)
    buff = io.BytesIO()
    pil_im.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    # img_str = base64.b64encode(img)
    # '''

    # FIXME: LOCAL TEST
    '''
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    pil_im = Image.fromarray(img_gray)

    buff = io.BytesIO()
    pil_im.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    # '''

    if result != None:
        return [str(result),str(img_str,'utf-8')]
    else:
        return [str(-1),str(img_str,'utf-8')]


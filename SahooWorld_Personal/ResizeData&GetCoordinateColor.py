import pandas as pd
import numpy as np
from PIL import Image

def resize96(trainingData):  # 전체 행의 image column을 96x96 행렬로 변환하여 resize_images에 입력
    resizeImages = []
    for i in range(len(trainingData.iloc[:, 0])):  # len(a.iloc([:, 0]) 은 행의 갯수를 추출하기 위해 만듬
        resizeImages.append(np.array(trainingData.iloc[i][30].split(" "), dtype=np.uint8).reshape(96, 96))
        # dtype를 uint8로 지정해줘야 그림이 출력됨
    return resizeImages


def getCoordinate(trainingData, row, column):
    # 좌표값을 찾는 함수
    # trainingData 매개변수는 csv파일 불러온 값.
    trainingData = trainingData
    row = row
    column = column
    coordX = 0
    coordY = 0
    if column == "left_eye_center":
        coordX = int(trainingData['left_eye_center_x'][row])
        coordY = int(trainingData['left_eye_center_y'][row])
    elif column == "right_eye_center":
        coordX = int(trainingData['right_eye_center_x'][row])
        coordY = int(trainingData['right_eye_center_y'][row])
    elif column == "left_eye_inner_corner":
        coordX = int(trainingData['left_eye_inner_x'][row])
        coordY = int(trainingData['left_eye_inner_y'][row])
    elif column == "left_eye_outer_corner":
        coordX = int(trainingData['left_eye_outer_corner_x'][row])
        coordY = int(trainingData['left_eye_outer_corner_y'][row])
    elif column == "right_eye_inner_corner":
        coordX = int(trainingData['right_eye_inner_corner_x'][row])
        coordY = int(trainingData['right_eye_inner_corner_y'][row])
    elif column == "right_eye_outer_corner":
        coordX = int(trainingData['right_eye_outer_corner_x'][row])
        coordY = int(trainingData['right_eye_outer_corner_y'][row])
    elif column == "right_eye_outer_corner":
        coordX = int(trainingData['right_eye_outer_corner_x'][row])
        coordY = int(trainingData['right_eye_outer_corner_y'][row])
    elif column == "left_eyebrow_inner_end":
        coordX = int(trainingData['left_eyebrow_inner_end_x'][row])
        coordY = int(trainingData['left_eyebrow_inner_end_y'][row])
    elif column == "left_eyebrow_outer_end":
        coordX = int(trainingData['left_eyebrow_inner_end_x'][row])
        coordY = int(trainingData['left_eyebrow_inner_end_y'][row])
    elif column == "right_eyebrow_inner_end":
        coordX = int(trainingData['right_eyebrow_inner_end_x'][row])
        coordY = int(trainingData['right_eyebrow_inner_end_y'][row])
    elif column == "right_eyebrow_outer_end":
        coordX = int(trainingData['right_eyebrow_outer_end_x'][row])
        coordY = int(trainingData['right_eyebrow_outer_end_y'][row])
    elif column == "nose_tip":
        coordX = int(trainingData['nose_tip_x'][row])
        coordY = int(trainingData['nose_tip_y'][row])
    elif column == "mouth_left_corner":
        coordX = int(trainingData['mouth_left_corner_x'][row])
        coordY = int(trainingData['mouth_left_corner_y'][row])
    elif column == "mouth_right_corner":
        coordX = int(trainingData['mouth_right_corner_x'][row])
        coordY = int(trainingData['mouth_right_corner_y'][row])
    elif column == "mouth_center_top_lip":
        coordX = int(trainingData['mouth_center_top_lip_x'][row])
        coordY = int(trainingData['mouth_center_top_lip_y'][row])
    elif column == "mouth_center_bottom_lip":
        coordX = int(trainingData['mouth_center_bottom_lip_x'][row])
        coordY = int(trainingData['mouth_center_bottom_lip_y'][row])
    return coordX, coordY


def getCoordinateColor(resizeImages, row, coordX, coordY):
    # 원하는 좌표값의 color값을 추출하는 함수
    # 매개변수 = (96x96로 변환된 이미지 리스트, 행값, 키 x값, 키 y값)
    return resizeImages[row][coordX - 1, coordY - 1]


def showImage(resizeImages, row):
    # 원하는 행의 사진을 출력합니다.
    # resizeImages.append(np.array(trainingData.iloc[i][30].split(" "), dtype=np.uint8).reshape(96, 96)) 에서
    # dtype를 uint8로 지정해줘야 그림이 출력됨
    Image.fromarray(resizeImages[row])


if __name__ == "__main__":
    trainingData = pd.read_csv("training.csv")

    resizeImages = []
    resizeImages = resize96(trainingData)

    coordX, coordY = getCoordinate(trainingData, 0, "left_eye_center")
    # getCoordinate의 3번째 매개변수 값은 아래에서 선택하면 된다.
    # ['left_eye_center', 'right_eye_center', 'left_eye_inner', 'left_eye_outer_corner', 'right_eye_inner_corner', 'right_eye_outer_corner',
    # 'left_eyebrow_inner_end', 'left_eyebrow_outer_end', 'right_eyebrow_inner_end', 'right_eyebrow_outer_end',
    # 'nose_tip', 'mouth_left_corner', 'mouth_right_corner', 'mouth_center_top_lip',
    # 'mouth_center_bottom_lip']

    print(getCoordinateColor(resizeImages, 0, coordX, coordY))

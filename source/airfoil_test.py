import torch

#%%
# model = torch.load('../result/airfoil-model.pth')
# model.eval()

# model = Net1()
# model.load_state_dict(torch.load('../result/airfoil-model.pth'))
# model.eval()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def reformat_image(ImageFilePath):

    image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]
    if(width != height):
        bigside = width if width > height else height

        background = Image.new('RGB', (bigside, bigside), (255, 255, 255, 255))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))

        width, height = image.size
        center1 = (int(0.5 * width), int(0.45 * height))
        center2 = (int(0.5 * width), int(0.55 * height))
        yellow = (255, 255, 0, 255)
        ImageDraw.floodfill(image, xy=center1, value=yellow)
        ImageDraw.floodfill(image, xy=center2, value=yellow)
        background.paste(image, offset)
        background = background.convert('1')
        background.save('out.jpg')
        print("Image has been resized !")

    else:
        print("Image is already a square, it has not been resized !")



reformat_image('/Users/udhayakumar/Desktop/Screenshot 2020-03-09 at 2.23.01 PM.png')

#%%
def rescale_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))
    image.save("out.jpg")
    image.show()


rescale_image('out.jpg')

#%%
def change_color():
    originalImage = cv2.imread('out.jpg')
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    plt.imshow(grayImage.reshape(int(np.sqrt(grayImage.size)), int(np.sqrt(grayImage.size))))
    plt.show()


# change_color()
#%%
def image_operation():
    image = cv2.imread('/Users/udhayakumar/Desktop/Screenshot 2020-03-09 at 2.23.01 PM.png')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #
    # # Remove horizontal
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    # detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    #
    # # Repair image
    # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    # result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    #
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('detected_lines', detected_lines)
    # cv2.imshow('image', image)
    # cv2.imshow('result', result)
    # cv2.waitKey()
    # inverse the image, so that lines are black for masking
    img = cv2.imread("/Users/udhayakumar/Desktop/Screenshot 2020-03-09 at 2.23.01 PM.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.bitwise_not(img)
    horizontal = th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    horizontal_inv = cv2.bitwise_not(horizontal)
    # perform bitwise_and to mask the lines with provided mask
    masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
    # reverse the image back to normal
    masked_img_inv = cv2.bitwise_not(masked_img)
    cv2.imshow("masked img", masked_img_inv)
    cv2.imwrite("result2.jpg", masked_img_inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# image_operation()

#%%
import pandas as pd
airfoil = pd.read_csv('/Users/udhayakumar/PycharmProjects/CNN-for-Airfoil/data/raw_data/csv/1_100.csv')

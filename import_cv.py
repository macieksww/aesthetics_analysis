import cv2 as cv
import os
from glob import glob

def process_directory(directory):
    subdirs = glob(directory+"/*/", recursive = True)
    print(subdirs)
    # number of images in the directory
    # needed to set proper batch_value for keras nn
    num_of_images = 0
    if len(subdirs) > 0:
        for subdir in subdirs:
            for filename in os.listdir(subdir):
                    filename = os.path.join(subdir, filename)
                    # checking if it is a file
                    if os.path.isfile(filename):
                        if filename.endswith('.jpg'):
                            num_of_images += 1
                            # img = cv.imread(filename)
                            # show image
                            # cv.imshow(filename, img)
                            # cv.waitKey(0)
    else:
        for filename in os.listdir(directory):
                filename = os.path.join(directory, filename)
                # checking if it is a file
                if os.path.isfile(filename):
                    if filename.endswith('.jpg'):
                        num_of_images += 1
    return num_of_images
                    
def rescale_images(img, scale=1):
    height = int(img.shape[0] * scale)
    width = int(img.shape[1] * scale)
    dimensions = (width, height)
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

def resize_to_paper_format(img):
    resized_image = cv.resize(img, (1240, 874), interpolation=cv.INTER_CUBIC)
    return resized_image

def blur_images(img, blur_coeffs=(3,3)):
    # blur_coeffs have to be odd numbers 
    # the higher the number, the more blurred the image is
    # may be useful to ignore letters and focus on the whole image
    # by the deep neural net
    blurred_img = cv.GaussianBlur(img, blur_coeffs, cv.BORDER_DEFAULT)
    return blurred_img

def convert_to_grayscale(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray_img

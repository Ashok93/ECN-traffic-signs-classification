import numpy as np
import cv2 # OpenCV computer vision library (https://opencv.org/)
import os # file operations
import math
import csv
import matplotlib.pyplot as plt # Plotting library (https://matplotlib.org/)

def get_train_images(img_path):

		images = []
		labels = []

		for c in range(0,43):
			prefix = img_path + '/' + format(c, '05d') + '/' # subdirectory for class
			gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
			gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
			next(gtReader) # skip header

			for row in gtReader:
				images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
				labels.append(row[7]) # the 8th column is the label
			gtFile.close()

		return images, labels

def get_test_images(img_path):

		images = []
		labels = []

		gtFile = open(img_path + '/GT-final_test.csv') # annotations file
		gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
		next(gtReader) # skip header

		for row in gtReader:
			images.append(plt.imread(img_path + '/' + row[0])) # the 1th column is the filename
			labels.append(row[7]) # the 8th column is the label
		gtFile.close()

		return images, labels


def preprocess_images(images, to_gray = True, size=(30,30)):
		processed_imgs = []
		for image in images:
			image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
			if to_gray:
				image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
				image = cv2.equalizeHist(image)

			norm_image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

			processed_imgs.append(norm_image)

		return processed_imgs

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .12+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    img = augment_brightness_camera_images(img)
    
    return img
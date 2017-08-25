import cv2
import numpy as np
import pytesseract

from PIL import Image


def get_processed_image(path):
    """Returns pre-processes image so that it is easier to read digits/characters from it."""
    
    # Read image
    img = cv2.imread(path)

    # Convert to grayscale and apply Gaussian filtering
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('t_gray.tif', img)

    # Resize image
    r = 500 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # cv2.imwrite('t_resize.tif', img)

    # Blur the image
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imwrite('t_blur.tif', img)

    # Threshold the image
    img_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # cv2.imwrite('t_thresh.tif', img_th)

    # Get connected components and use them to remove noise
    maxArea = 150
    minArea = 10
    
    comp = cv2.connectedComponentsWithStats(img_th)

    labels = comp[1]
    labelStats = comp[2]
    labelAreas = labelStats[:,4]    

    for compLabel in range(1,comp[0],1):

        if labelAreas[compLabel] > maxArea or labelAreas[compLabel] < minArea:
            labels[labels == compLabel] = 0

    labels[labels>0] =  1

    # Perform dilation on the image
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    IdilateText = cv2.morphologyEx(labels.astype(np.uint8),cv2.MORPH_DILATE,se)

    img_inv = (255-img_th)
    cv2.imwrite('temp.tif', img_inv)

    return img_inv

def extract_characters(image):
    """Return a list of images around the characters/digits found.
        This function does so by using contours to put characters/digits into seperate boxes"""

    # Find contours in the image
    _, ctrs, hier = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles containing each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects = sorted(rects, key=lambda rect: rect[0] + 5*rect[1])

    images = []
    for rect in rects:
        x1, y1, w, h = rect

        # Remove small contours
        if h < 20:
            continue
        
        x2 = x1 + w
        y2 = y1 + h
        
        images.append(img[y1:y2, x1:x2])

    # Save character images if necessary
    for k, i in enumerate(images):
        cv2.imwrite('t_' + str(k) + '.png', i)

    return images

def perform_ocr(image):
    return pytesseract.image_to_string(Image.open('temp.tif'), config="-c tessedit_char_whitelist=0123456789 --psm 12 --oem 0")


img_name = 'phone3.png'
img = get_processed_image('./images/' + img_name)
images = extract_characters(img)

print('Number of characters found:', len(images))
print('OCR Result:', perform_ocr(img))


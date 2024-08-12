import numpy as np
import cv2
from PIL import Image

def straighten_license_plate(license_plate_img):
    # Convert the image to OpenCV format (BGR)
    license_plate_cv = cv2.cvtColor(np.array(license_plate_img), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(license_plate_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for denoising
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny filter
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Find the license plate contour
    license_plate_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            license_plate_contour = approx
            break

    if license_plate_contour is None:
        raise ValueError("License plate contour not found")

    # Apply perspective transform to straighten the license plate
    pts = license_plate_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(license_plate_cv, M, (maxWidth, maxHeight))
    
    # Convert back to PIL image format
    straightened_license_plate = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    return straightened_license_plate


import pytesseract


def extract_text_from_image(image):
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image, config='--psm 8')  
    return text.strip()

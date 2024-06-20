import cv2
from skimage.metrics import structural_similarity as ssim
import imutils
import numpy as np

def take_picture(filename):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open video device")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cv2.imshow('Press Space to capture', frame)
    while True:
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(filename, frame)
            break
        ret, frame = cap.read()
        cv2.imshow('Press Space to capture', frame)
    cap.release()
    cv2.destroyAllWindows()

def load_and_resize(image_path, size=(640, 480)):
    image = cv2.imread(image_path)
    return cv2.resize(image, size)

def compare_images(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Original", imageA)
    cv2.imshow("Modified", imageB)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reference_image_path = "reference.jpg"
    selfie_image_path = "selfie.jpg"

    # Take a reference image
    print("Take the reference image.")
    take_picture(reference_image_path)

    # Take a selfie image
    print("Take the selfie image.")
    take_picture(selfie_image_path)

    # Load and resize images
    reference_image = load_and_resize(reference_image_path)
    selfie_image = load_and_resize(selfie_image_path)

    # Compare images
    compare_images(reference_image, selfie_image)

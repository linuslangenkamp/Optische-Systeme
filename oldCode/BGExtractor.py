import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

def extractBG(img):
    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    # Define the rectangle around the object (foreground)
    # rect = (50, 30, img.shape[1] - 80, img.shape[0] - 50)
    rect = (10, 10, img.shape[1] - 30, img.shape[0] - 30)
    # Initialize the background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    # Apply GrabCut algorithm
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    # Modify the mask to create a binary mask for the foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # Apply the mask to the original image
    result = img * mask2[:, :, np.newaxis]
    return result


if __name__ == "__main__":
    # Load the image
    for i in range(500, 3000):
        image_path = r'C:\Users\Linus\Desktop\Studium\Master\Optische Systeme\Projekt\archive\asl_alphabet_train\asl_alphabet_train\A\A%s.jpg' % i
        img = cv2.imread(image_path)
        result = extractBG(img)

        # Display the original image, mask, and result
        """
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.imshow(img_rgb), plt.title('Original Image')
        plt.subplot(132), plt.imshow(mask, cmap='gray'), plt.title('GrabCut Mask')
        plt.subplot(133), plt.imshow(result), plt.title('Foreground Extraction')
        plt.show()
        """
        cv2.imshow("Image", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

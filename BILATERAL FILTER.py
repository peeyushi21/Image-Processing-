import cv2
import numpy as np

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    no_flash_image=cv2.imread(image_path_a)
    flash_image = cv2.imread(image_path_b)
    def check(img):
        hist = cv2.calcHist(flash_image, [0], None, [256], [0, 255])
        threshold = 5  # Adjust this threshold value as needed
        exceeding_indices = np.where(hist > threshold)[0]

        n=exceeding_indices.shape[0]
        if(n>150):
            ans=2
        elif(n>60):
            ans=3
        else:
            ans=1
        return ans
    
    ans=check(no_flash_image)
    if(ans==2):
        s1=50
        s2=30
    elif(ans==3):
        s1=10
        s2=5
    elif(ans==1):
        s1=2
        s2=0.5
    
    def bilateral_filter(image: np.ndarray, kernel_size: int, sigma: int, sigma_r: int):
     height, width, channels = image.shape

     pad_height = kernel_size // 2
     pad_width = kernel_size // 2
     padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_REFLECT)

     output_image = np.zeros_like(image)

     x, y = np.meshgrid(np.arange(-pad_width, pad_width + 1), np.arange(-pad_height, pad_height + 1))
     gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

     for i in range(height):
        for j in range(width):
            for k in range(channels):
                # Extract the local region
                region = padded_image[i:i+kernel_size, j:j+kernel_size, k]

                # Calculate the range kernel
                range_kernel = np.exp(-(region - image[i, j, k])**2 / (2 * sigma_r**2))

                # Compute the bilateral filter response
                kernel = range_kernel * gaussian_kernel
                normalized_kernel = kernel / np.sum(kernel)
                output_image[i, j, k] = np.sum(region * normalized_kernel)

     return output_image

    def denoise_image(img, sigma_space, sigma_range):
     img = np.array(img).astype(float)
     Sigma = sigma_space
     Sigma_r = sigma_range
     result=bilateral_filter(img, kernel_size=21, sigma=Sigma, sigma_r=Sigma_r)
     return result.astype(np.uint8)
    
    Fd = denoise_image(flash_image,s1,s2)
    Ad = denoise_image(no_flash_image,s1,s2)
    e = 0.02

    result = Ad * ((flash_image.astype(np.float32) + e) / (Fd.astype(np.float32) + e))
    return result

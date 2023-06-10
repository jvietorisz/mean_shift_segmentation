'''
Utilities for image segmentation by mean shift.
Written by Jacob Sage Vietorisz
Latest Update: June 8, 2023

These functions are called in the main script of the project, main.py
'''
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_image(image_path, resolution):
    '''
    Imports and prepares an image for processing.

    :params:
    image_path: string. path to the image file
    resolution: tuple. (M, N), the desired dimensions, of the output image (MxN)
    
    :returns:
    image: MxNx3 np.ndarray. The array representation of an image with values on [0,1) as 32 bit integers
    '''
    from skimage import io, img_as_float32
    from skimage.transform import resize
    
    # Load image from path
    image = io.imread(image_path)

    # Check for grayscale
    if len(image.shape) == 2:
        raise ValueError('This implementation does not support grayscale images')

     # Adjust datatype if necessary
    if np.dtype(image[0,0,0])=='float32' or np.dtype(image[0,0,0])=='float64':
        # image = img_as_uint(image)
        pass
    if np.dtype(image[0,0,0])=='uint8' or np.dtype(image[0,0,0])=='uint16':
        image = img_as_float32(image)
    else:
        print(f'WARNING: Unrecognized data type in image. The datatype is {np.dtype(image[0,0,0])}')
    
    # Resize image to specified resolution
    image = np.floor(255*resize(image, resolution, anti_aliasing=True))
    # rescale values to [0, 255]
    image = np.uint8(image)
    
    return image


def extract_features(image):
    '''
    Extracts the feature of each pixel in an image. 
    The feature space is five dimensional and depends on the i,j indices 
    (equivalently x, y coordinates) of the image, as well as its three color channels, R, G, and B.

    :params:
    image: MxNx3 np.ndarray. The image to be segmented.

    :returns:
    features: (M*N)x5 np.ndarray. Matrix of features whose number of principal elements 
              is the number of pixels in image.
    '''
    
    from skimage.filters import sobel
    
    image_height = image.shape[0]
    image_width = image.shape[1]

    grayscale_image = 0.2125*image[:,:,0] + 0.7154*image[:,:,1] + 0.0721*image[:,:,2]
    sobel_img = sobel(grayscale_image)

    # Construct feature vectors for each pixel
    features = np.zeros((image_height*image_width, 5))
    for i in range(image_height):
        for j in range(image_width):
            # the 5 coordinates for each pixel feature vector are the pixel positions and the RGB values
            features[i*image_width + j] = np.array([0.05*i, 0.05*j, *image[i,j,:]], )

    return features


def mean_shift(features, window_size, epsilon, iterations):
    '''
    Performs the mean shift algorithm on each pixel of an image in parallel to find the modes of the data.

    :params: 
    features: (M*N)x5 np.ndarray. Matrix of features whose number of principal elements 
              is the number of pixels in image.
    window_size: float. The distance in feature space within which to sample features
                to calculate new mean.
    epsilon: float. The threshold below which a mean shift is neglected and a mode is assigned

    :returns:
    updated_means: (M*N)x5 np.ndarray. Matrix of mode features. Each element is the vector in feature space
                   to which each pixel converges during mean shift.
    '''
    # Initialize the means to the features for each pixel
    updated_means = features
    
    # Initialize counter
    for i in tqdm(range(iterations), desc='Performing mean shift...'): # Use progress bar to display worst-case progress
        # Save the means for comparison after update
        previous_means = updated_means
        
        # Parallelize distance computation
        B = 2*np.matmul(updated_means, np.transpose(features))
            
        magF1_sq = np.sum(np.power(updated_means, 2), axis=1, keepdims=True) 
        magF2_sq = np.sum(np.power(features, 2), axis=1, keepdims=True)
        A = magF1_sq + np.transpose(magF2_sq)

        D = np.sqrt(A - B) # D is the vector of distances between the mean vector and all of the features   
        # Convert D to binary matrix
        D[D>window_size]=0 # kill distances outside of neighborhood defined by window size
        D[D>0]=1 # Set nonzero distances to 1 to evenly weight included features

        # Calculate new means
        mags_of_D_rows = np.sum(D, axis=1, keepdims=True)
        mags_of_D_rows[mags_of_D_rows==0]=1
        sum_of_features_in_neighborhood = np.linalg.multi_dot([D,features])
        new_means = np.divide(sum_of_features_in_neighborhood, mags_of_D_rows)

        # Calculate distances from previous means to new means
        distance_to_new_means = np.linalg.norm((updated_means - new_means), axis=1)
        distance_to_new_means[:, np.newaxis] # Add axis and reshape so distance_to_new_means.shape[0] == means.shape[0]

        # Update each mean if the distance to new mean is greater than threshold
        cond1 = np.expand_dims(distance_to_new_means > epsilon, axis=1)
        cond2 = np.expand_dims(distance_to_new_means <= epsilon, axis=1)

        updated_means =  np.select(condlist = [cond1, cond2], choicelist = [new_means, updated_means])

        # Compare with previous means
        if np.array_equal(updated_means, previous_means):
            break; # If no change this iteration, break loop

        if i==iterations-1:
            print('Maximum mean shift iterations performed.')

    return updated_means


def segment_image(image, updated_means):
    '''
    Displays the orignial image along with its mean shift segmentation.

    :params:
    image: MxNx3 np.ndarray. The image to be segmented.
    updated_means: (M*N)x5 np.ndarray. Matrix of mode features. Each element is the vector in feature space
                   to which each pixel converges during mean shift.

    :returns:
    segmented_image: MxNx3 np.ndarray with dtype float64. The modified image where each pixel is replaced by the RGB values 
                     of its mean vector. Pixels that converge to the same mean vector (mode) are represented 
                     by the same color in the segmented image.
    '''
    segmented_image = np.zeros((image.shape))

    # Populate segmented image
    for n in range(len(updated_means)):
        i = int(n//image.shape[1])
        j = int(n%image.shape[1])
        R = updated_means[n][2]
        B = updated_means[n][3]
        G = updated_means[n][4]
     
        segmented_image[i, j, :] = [R,B,G]
    segmented_image = np.floor(segmented_image)
    segmented_image = np.uint8(segmented_image)
    
    return segmented_image


def visualize_results(image, segmented_image):
    '''
    Plots the original and segmented images after adjusting their data type to uint8 on [0, 255].

    :params:
    image: MxNx3 np.ndarray. The image to be segmented.
    segmented_image: MxNx3 np.ndarray with dtype float64. The modified image where each pixel is replaced by the RGB values 
                     of its mean vector. Pixels that converge to the same mean vector (mode) are represented 
                     by the same color in the segmented image.
    :returns:
    None
    '''
    from skimage import img_as_ubyte
    
    # Adjust datatypes
    # image = img_as_ubyte(image)
    # segmented_image = img_as_ubyte(segmented_image)
    
    
    # Plot images
    fig, ax = plt.subplots(nrows=1, ncols=2)
    # Plot the origninal image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    # Plot mean shift segmented image
    ax[1].imshow(segmented_image)
    ax[1].set_title('Segmented Image')
    
    plt.show()
    return

def visualize_difference(image, segmented_image):
    
    from skimage import img_as_ubyte

    # Plot images
    difference = image - segmented_image

    if np.min(difference)<0:
        difference = difference+np.abs(np.min(difference))
        difference = difference/np.max(difference)
    else:
        difference = difference/np.max(difference)

    plt.imshow(difference)
    plt.show()
    return
    

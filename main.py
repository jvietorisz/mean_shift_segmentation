'''
Mean Shift Segmentation Implementation
By Jacob Sage Vietorisz
Latest update: June 8, 2023

Written for rStream Recycling as a Machine Learning Researcher Summer 2023. 
This implementation takes in an RGB image and performs mean shift segmentation.
'''
# Import necessary libraries
from utilities import prepare_image, extract_features, mean_shift, segment_image, visualize_results, visualize_difference

def main():
    # Define image path and resolution
    image_path = './images/cat_hard.jpeg'
    resolution = (100, 100)

    # Create image instance
    image = prepare_image(image_path, resolution)

    # Extract the features of each pixel
    features = extract_features(image)

    # Define mean shift parameters and perform mean shift to find modes of the data
    window_size = 40
    epsilon =  0.2
    iterations = 100
    updated_means = mean_shift(features, window_size, epsilon, iterations)

    # Assign each pixel in the original image to the RGB values of their mean vector mappings
    segmented_image = segment_image(image, updated_means)

    # Visualize the results
    visualize_results(image, segmented_image)
    return

if __name__ == '__main__':
    main()





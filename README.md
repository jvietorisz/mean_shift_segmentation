# mean_shift_segmentation
I wrote a Python implementation of the mean shift algorithm for performing image segmentation. 

I completed this project as a warmup assignment for my role as a computer vision summer research intern at rStream Recycling. Segmentation is a useful approach to waste sorting because it enables simultaneous classification and localization, provided that the camera geometry is known. State-of-the-art semantic and instance segmentation techniques use neural netorks to segment images, but mean shift is a very popular classical approach to the image segmentation problem – which made it a worthwhile exercise to implement myself!

The algorithm works by extracting a simple feature vector for each pixel in the image. In my implementation, the features are 5-dimensinoal vectors composed from pixel's RGB values and down-scaled positional information from its array indices.

For each feature (pixel), the distance to all other features is calculated in feature space, and all of the features within a specified distance from the feature of interest are averaged to find a regional mean. The feature of interest is then shifted to that mean, and the process is repeated, calculating the distances to all of the features, and averaging the features within a specified distance of the mean. This iterative approach pushes pixel features to converge to the modes of the data distribution, which are the regions in feature space with the highest density of features.

In a naive implementation using loops, this is extremely computationally expensive. I resolved this issue by vectorizing my implementation, so that every iteration of shifting the mean is performed at once for all pixels in the image. Check out the project page on my website – jacobsagevietorisz.com – for a more detailed explanation!

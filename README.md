# MATH-651-Project
Place for our project.

## Introduction
With the development of sciences and technology, we have a growing number of high resolution satellite images.
This help us get a better understanding of the Earth.
Labelling feathers such us water, farm, roadway, artificial structure, etc, in satellite images has been became an important task.
At present, this labelling work is still mainly done by hand and semi-automated algorithms.
With more and more complex image data we get, a reliable automated algorithm was needed.

Under this background, the Defence Science and Technology Laboratory (Dstl) is seeking novel solutions to alleviate the burden on their image analysts and they start up a competition on Kaggle.com.
The datasets the provided is containing three band remote sensing images, sixteen band remote sensing images, object locations and types.
All the datasets can be used for the labelling algorithm.

## Objectives
Our objective is to come up with a work flow for label objects types on satellite images.
First, we will study image edge detection algorithms, comparing different algorithms with satellite images.
Then, by using unsupervised learning algorithm, we will try to identify the road, track, waterway, standing water, etc types.
Finally, we will conduct convolutional neural network to identify buildings in the images.

## Edge detection algorithms
Edge detection is an essential component in many computer vision problems.

## Convolutional neural network

## Management Plan
There are some separated work:
1. Visualization module: show the image and the object polygon.
2. Edge detection algorithm: mainly based on the Canny edge detector.
3. Unsupervised learning method: such as k-means and SVM, generate a pipline for analysis satellite images.
4. Convolutional neural network: we may can consider using polygon file directly as a training set.

## Conclusion

## References
- Canny J. A computational approach to edge detection[J]. IEEE Transactions on pattern analysis and machine intelligence, 1986 (6): 679-698.
- Lindeberg T. Edge detection and ridge detection with automatic scale selection[J]. International Journal of Computer Vision, 1998, 30(2): 117-156.
- Mallat S. A wavelet tour of signal processing: the sparse way[M]. Academic press, 2008.

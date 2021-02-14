# Project: Depth of Field Prediction from RAW images

This repo contains a single Jupyter Notebook (Python) for the Machine Learning Course at the MSc Program of AI, organized by NCSR Demokritos and University of Piraeus.

Instructor: Mr. Theodoros Giannakopoulos - [tygiannak](https://github.com/tyiannak)

## Notebook Sections

### Imports

* [matplotlib](https://github.com/matplotlib/matplotlib), [plotly](https://github.com/plotly) and [seaborn](https://github.com/mwaskom/seaborn) for Visualization

* [numpy](https://github.com/numpy/numpy) for Matrices

* [opencv](https://github.com/opencv/opencv) for Computer Vision

* [sklearn](https://github.com/scikit-learn/scikit-learn) for Machine Learning

* [keras](https://github.com/keras-team/keras) for Deep Learning

### Data Collection

* Toy Dataset from personal portofolio used for *Data Exploration*
* Main Dataset download from [sniafas/photography-style-analysis](https://drive.google.com/file/d/1Eht4PDWlRWhWalWXP53da8CvaWy1TBEx/view)

### Data Preparation & Pre-processing

* Handcrafted preparation of the Training/Test datasets. 500 images labeled as Deep(0) or Shallow(1) DoF.
* 80/20 ratio (Training: 400 / Test: 100)
* Image Standardization - resize images to 100x100

### Data Exploration

* Changing Colorspaces (GBR, RGB, Grayscale)
* Geometric Transformations (Scale, Rotate)
* Smoothing (Blur, GaussianBlur)
* Thresholding
* Edge Detection
* Morphological Transformations (Open, Gradient)
* Histograms (RGB, Grayscale, Edges)
* Gradients (Sobel, Laplacian)

![alt text](https://imgur.com/wsDnZha.png)
![alt text](https://imgur.com/JOQSzjZ.png)
![alt text](https://imgur.com/zU214dY.png)
![alt text](https://imgur.com/q4kdEoI.png)
![alt text](https://imgur.com/3lqrA8k.png)
![alt text](https://imgur.com/oai2EYM.png)

### Feature Extraction

* For kNN, SVC:
  * Load images as 16-bit
  * Remove noise by blurring with a Gaussian filter
  * Convert the image to grayscale
  * Apply Laplace function
  * Convert images back to 8-bit
  * The output of this process is a features vector

* For CNN there is no need for Feature Extraction

![alt text](https://imgur.com/neF0nwD.png)
![alt text](https://imgur.com/p2yxWFL.png)
![alt text](https://imgur.com/2XGtNCi.png)

### Data Revision and Augmentation

After experimenting with the initial dataset:

* Consider more data (via image transformation - rotation) from 200 images to 500
* Reconsider image size from 100x100 back to original size (200x200)

### kNN Implementation

#### Results

Mean and Standard Deviation Averages of various Metrics on k ranges 1-15

![alt text](https://imgur.com/SNe4FJg.png)

### SVC Implementation

#### Train/Test Accuracy VS Model Complexity

![alt text](https://imgur.com/hM6YkfU.png)

#### Results

![alt text](https://imgur.com/jDlGCXa.png)

#### Precision-Recall & ROC Curves

![alt text](https://imgur.com/AC354ws.png)

### CNN Implementation

That's a secret...

### Discussion

Problems during implementation:
* DoF is a concept closely related to the 3D world and trying to identify it inside a 2D image with traditional ML methods proved challenging
* DoF estimation (for labeling) in 200x200 size images was eye-hurting
* OpenCV library's default colorspace is BGR. Extra transformations needed to plot
* Bias -> shallows are humans/animals/plants, deeps are landscapes and many are from top down angle
* White background TOP DOWNs and deep DoF images which include sky and/or sea reflection were becoming a noise for our Deep dataset after the Laplacian Filter 

### Future Work

Experiment with:

* Different libraries like Mahotas, which provides more advanced features such as haralick, local binary patterns, etc
* Different algorithms, like Decision Trees (..but, CNN FTW)
* RGB, HSV instead of Grayscale
* Bigger Dataset (1000+ images)
* Bigger image size (maybe 600x600)
* SMOTE for Dataset’s balancing

## Authors

* **Michael Zouros** - [mzouros](https://github.com/mzouros)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Theodoros Giannakopoulos - [tygiannak](https://github.com/tyiannak)
* Steve Niafas - [sniafas](https://github.com/sniafas)
* Ming Tang - [tangming2008](https://github.com/tangming2008)


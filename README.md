# Project: Dog Breed Classifier
## Category: Deep Learning

[//]: # (Image References)

[image1]: ./images/my-output-sample.jpg "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


### Project Overview

In this project, I learned how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, my algorithm identifies an estimate of the canine's breed. If supplied an image of a human, the code identifies the resembling dog breed.

The main part of this project is exploring Convolutional Neural Networks(CNN) models for classification. Not only about the classification algorithm architecture, I also had to make decisions about the user experience for this app. Through this project, I discovered the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.

### App Design

This app detects dog breeds from the input image.

- If there is a dog in the image, the app shows the detected breed of the dog.
- If there is a person in the image, the app shows the resemble dog breed.
- This app displays an image of the predicted breed as well as the name of the breed.
- If there is neither a dog nor a person in the image, the app returns an error message.
- This app estimates up to three most likely dog breeds. When it is likely to be a single breed, it returns just one breed. It suggests multiple breeds if the dog is predicted to be a mixed-breed.  

![Sample Output][image1]


### Techniques

- Human face detection using [OpenCV: Haar Feature-based Cascade Classifiers](https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)
- Dog image detection using a pre-trained [ResNet-50 model from Keras Applications](https://keras.io/applications/#resnet50)
- Building a CNN model from scratch
- Compile, train, and test the model
- Data augmentation
- Transfer learning (using bottleneck features of [Xception](https://arxiv.org/abs/1610.02357))


### Libraries

This project requires **Python 3.** and the following libraries installed:

- [Numpy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)

This project runs on the [Jupyter Notebook](http://ipython.org/notebook.html)

### Files

- `dog_app.ipynb`: Jupyter notebook, This project's main file
- `dog_app.html`: HTML, Output of the project (for submission)
- `extract_bottleneck_features.py`: Python module to extract bottleneck features
- `images/`: Image files shown in Jupyter Notebook and README
- `my_images/`: Image files used for the test
- `haarcascades/haarcascade_frontalface_alt.xml`: Pre-trained haar-cascade face classifier with OpenCV (Usage: [Face Detection using Haar Cascades](https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html) )
- `requirements/`: Required packages for installing (see below)

### Data

Dataset files and bottleneck features below has to be downloaded to run the app.

- [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) - unzip and place at `/data/dog_images`
- [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) - unzip and place at `/data/lfw`
- [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) - place at `/data/bottleneck_features/`
- [Xception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) - place at `/data/bottleneck_features/`

These files are downloaded from the link, and placed under the directory specified above. Or you could change the corresponding code in Jupyter Notebook so that it can load this dataset from the directory where you placed the dataset. Note that the directory structure of each dataset must not be changed.



### Install

**GPU support machine** is desirable to run this project.

1. __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

2.  **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`):
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`):
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

3. **If you are running the project on your local machine (and not using AWS)** and Step 2 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```

4. **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__:
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__:
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

6. **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment.
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```
7. Download and place datasets in proper directories.

8. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.


### Note
This project has been done as a part of Machine Learning Engineer Nanodegree program, at [Udacity](https://www.udacity.com/).

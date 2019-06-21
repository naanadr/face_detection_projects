# Facial_Expression_Recognition


* To run this project:

  `$ docker run --rm -it --name=facial-image -v $PWD/expression_recog:/app/Workspace fernandaszadr/facial-expression-recognition`

* To run this project and use `cv2.imshow` (Linux user):

  `$ xhost +`

  `$ docker run --rm -it --name=facial-image --net=host --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v $PWD/expression_recog:/app/Workspace fernandaszadr/facial-expression-recognition`

* Label used to represent classes in JAFEE and CK+ dataset:
  * 0 -> Angry
  * 1 -> Disgust
  * 2 -> Fear
  * 3 -> Happy
  * 4 -> Neutral
  * 5 -> Sad
  * 6 -> Surprise

----

### Steps executed in this project:
  * Detected face with Viola Jones (Haar classifier) implemented in OpenCV;
  * Apply histogram normalize;
  * Extract shape informations with Zernike Moments;
  * Extract texture information with Local Binary Patterns;
  *

---

### References:
  * [Local Binary Pattern in Scikit-Image](https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern)
  * [SVM in Scikit-Lean](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
  * [Building a Pokedex in Python: Indexing our Sprites using Shape Descriptors](https://www.pyimagesearch.com/2014/04/07/building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/)
  * [OpenCv histogram equalization](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html)
  * [Local Binary Patterns with Python & OpenCV](https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)
  * [Texture Matching using Local Binary Patterns (LBP), OpenCV, scikit-learn and Python](http://hanzratech.in/2015/05/30/local-binary-patterns.html)
  * [Face Detection using Haar Cascades ](https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0)
  * [Object Detection : Face Detection using Haar Cascade Classfiers](https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php)
  * [Face Detection with Python using OpenCV](https://www.datacamp.com/community/tutorials/face-detection-python-opencv)
  * [(Faster) Facial landmark detector with dlib](https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/)
  * [Real-time facial landmark detection with OpenCV, Python, and dlib](https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/)
  * [Facial landmarks with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
  * [Data augmentation : boost your image dataset with few lines of Python](https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec)
  * [Neural network models (supervised) Scikit-Learn](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

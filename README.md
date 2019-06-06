# Facial_Expression_Recognition


* To run this project:

  `$ docker run --rm -it --name=facial-image -v $PWD/expression_recog:/app/Workspace fernandaszadr/facial-expression-recognition`

* To run this project and use `cv2.imshow` (Linux user):

  `$ xhost +`

  `$ docker run --rm -it --name=facial-image --net=host --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v $PWD/expression_recog:/app/Workspace fernandaszadr/facial-expression-recognition`

* Label used to represent classes:
  * 0 -> Angry
  * 1 -> Disgust
  * 2 -> Fear
  * 3 -> Happy
  * 4 -> Neutral
  * 5 -> Sad
  * 6 -> Surprise

### Steps executed in this project:
  * Detected face with Viola Jones (Haar classifier) implemented in OpenCV;
  * 

# Facial_Expression_Recognition

To run this project:

`docker pull opencv_FER`

and

`docker run -it opencv_FER`

#### Primeiro:

`docker build --no-cache -t fernandaszadr/facial-expression-recognition .`
`docker run -it facial-fer fernandaszadr/facial-expression-recognition`

ou

`docker run -it facial-fer -v path_local:path_docker fernandaszadr/facial-expression-recognition`

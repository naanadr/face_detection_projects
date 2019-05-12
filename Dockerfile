# version 0.1

FROM python:3.7
RUN apt-get update -y
RUN apt-get install build-essential cmake  -y
RUN apt-get install libgtk-3-dev  -y
RUN apt-get install libboost-all-dev  -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
ADD . /work
WORKDIR /work
CMD /bin/bash

FROM python:3.6

WORKDIR /app

# for dlib
RUN apt-get update && apt-get -y install cmake && apt-get install bzip2

# model for landmark detection
RUN wget -O predictor.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
RUN bzip2 -d  predictor.dat.bz2

ENV PREDICTOR_PATH=./predictor.dat

# install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --cache-dir=/cache -r requirements.txt

#copy project directory
COPY ./ .


ENTRYPOINT ["python", "./main.py"]
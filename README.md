# test-task-EORA
## Pavel Nikulin
p.nikulin@innopolis.ru

tg: @indionapolis

this repository contains implementation of test task on head pose estimation on video for summer internship ata EORA.

## Getting Started

if you want to run this project locally follow instructions in Dockerfile

### Prerequisites

you need ```docker```

### Installing

just clone this directory and run ```boot.sh```

```
git clone https://github.com/indionapolis/test-task-EORA.git
```

And start project (specify ip address of camera to get video stream from ```STREAM_URL=<url>``` in ```boot.sh```)

```
sh boot.sh
```

then go to [http://localhost](http://localhost) and you will be able to see processed video stream

## Built With

* [python](https://www.python.org) - programming language
* [OpenCV](https://opencv.org) - library of programming functions mainly aimed at real-time computer vision
* [dlib](http://dlib.net) - toolkit containing machine learning algorithms
* [flask](https://flask.palletsprojects.com/en/1.1.x/) - web framework


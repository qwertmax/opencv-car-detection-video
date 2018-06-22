# OpenCV Car detection from video stream

## How to run the code
```bash
git clone git@github.com:qwertmax/opencv-car-detection-video.git
cd opencv-car-detection-video
python main.py
```

## install packages

```bash
pip install -r requirements.txt
```


## Example how it works

[![OpenCV Car detection from video stream](http://img.youtube.com/vi/RCFkCmPqJrk/0.jpg)](https://youtu.be/RCFkCmPqJrk)

[![OpenCV Car detection from video stream](http://img.youtube.com/vi/KiAl5VfvKX4/0.jpg)](https://youtu.be/KiAl5VfvKX4)

## python code

```python
import numpy as np
import cv2

cascade_src = 'cars.xml'
video = 'data/video1.avi'


def detectCars(filename):
  rectangles = []
  cascade = cv2.CascadeClassifier(cascade_src)

  vc = cv2.VideoCapture(filename)

  if vc.isOpened():
      rval , frame = vc.read()
  else:
      rval = False


  while rval:
    rval, frame = vc.read()
    frameHeight, frameWidth, fdepth = frame.shape

    # Resize
    frame = cv2.resize(frame, ( 600,  400 ))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # haar detection.
    cars = cascade.detectMultiScale(gray, 1.3, 3)


    for (x, y, w, h) in cars:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


    # show result
    cv2.imshow("Result",frame)

    if cv2.waitKey(33) == ord('q'):
      break

  vc.release()


detectCars(video)
```

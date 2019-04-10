# Face classification and detection [FACE RECO].
Real-time face detection and emotion/gender classification using fer2013/IMDB datasets with a keras CNN model and openCV Haar Cascades.

## Created By:

Universidad de San Buenaventura Cali (LIDIS Team).
* Andrés Felipe Girón - [anfegiar@gmail.com]
* Juan Pablo Chacón -   [juanphax7@gmail.com]

## Training models:
* IMDB gender classification test accuracy: 96%.
* fer2013 emotion classification test accuracy: 66%.

Based on the [B-IT-BOTS robotics team](https://mas-group.inf.h-brs.de/?page_id=622) original system.

You have to move to the src/webII folder, where the real system is.
In the file "camera.py" is the image capture throug the service.
In the "makeup-artist" is where all the logic is.


#To serve throug Waitress server:

> waitress-serve --listen=*:8000 app:app

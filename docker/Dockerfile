FROM tensorflow/tensorflow:latest-gpu-py3

# Update apt-get and get the necessary cv2 packages
RUN apt-get update \
&& apt-get install -y libsm6Z libxext6 libxrender-dev
RUN pip install opencv-python tqdm efficientnet
RUN git clone https://github.com/calciver/EfficientNet_Transfer_Learning.git

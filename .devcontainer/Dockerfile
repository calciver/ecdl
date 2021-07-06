# FROM tensorflow/tensorflow:latest-gpu

# Update apt-get and get the necessary cv2 packages
# RUN apt-get update \
# && apt-get install -y libsm6Z libxext6 libxrender-dev
# RUN pip install opencv-python tqdm efficientnet
# RUN git clone https://github.com/calciver/EfficientNet_Transfer_Learning.git
# Get Tensorflow base image
FROM tensorflow/tensorflow:latest-gpu

# MAINTAINER russ chua 

RUN apt-get update && apt-get install -y git

RUN python3 -m pip install efficientnet
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install jupyter notebook
# RUN git clone https://github.com/calciver/ecdl.git

# CMD ["echo", "TF2 Brain Segmentation Docker Image"]

ENV PATH="/workspaces/ecdl:${PATH}"
ENV PYTHONPATH="/workspaces/ecdl:${PYTHONPATH}"
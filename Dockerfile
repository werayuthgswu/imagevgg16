FROM python:3.7-bullseye

# install build utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get -y upgrade

# clone the repository 
#RUN git clone --depth 1 https://github.com/tensorflow/models.git

# Install object detection api dependencies
# Install object detection api dependencies
#RUN apt-get install -y protobuf-compiler && \
RUN pip install Cython && \
    pip install contextlib2 && \
#    pip install jupyter && \
#    pip install matplotlib && \
    pip install pycocotools && \
    pip install opencv-python && \
    pip install flask && \
    pip install tensorflow && \
    pip install Pillow && \
    pip install tf_slim && \
    pip install tk && \
    pip install lxml && \
    pip install pillow && \
    pip install requests

# # Get protoc 3.0.0, rather than the old version already in the container
#RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
#    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \

#COPY proto3/bin/* /usr/local/bin
#COPY proto3/include/* /usr/local/include

#RUN cd models/research && \
#    protoc object_detection/protos/*.proto --python_out=.

# Set the PYTHONPATH to finish installing the API
ENV PYTHONPATH=$PYTHONPATH:/models/research/object_detection
ENV PYTHONPATH=$PYTHONPATH:/models/research/slim
ENV PYTHONPATH=$PYTHONPATH:/models/research

# -p /home/tensorflow
COPY tensorflow.tar /home/
WORKDIR /home
RUN tar -xvf tensorflow.tar
RUN rm -f tensorflow.tar
#RUN rm -f tensorflow/.git
WORKDIR /home/tensorflow

RUN pip install google-api-python-client google-auth

RUN pip uninstall -y protobuf
RUN pip install protobuf==3.19.6

CMD ["python", "main.py"]


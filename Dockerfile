FROM nvcr.io/nvidia/tensorrt:21.06-py3

RUN apt-get update && apt-get upgrade -y && apt-get -y install gosu vim 

RUN apt-get install -y libgl1-mesa-dev

RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install colorama
RUN pip install pillow
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install pyyaml
RUN pip install argparse
RUN pip install tqdm
RUN pip install tensorflow

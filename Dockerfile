FROM nvidia/cuda:11.0-devel-ubuntu18.04

MAINTAINER kmj km_jin@korea.ac.kr

ENV LANG C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3-setuptools python3-wheel

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt update && apt install gcc-9 bison gawk -y
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

RUN python3.6 -m pip install pip --upgrade

RUN mkdir -p /OTPose
WORKDIR /OTPose

ADD environment.yaml /OTPose
RUN conda update conda && conda env create --file environment.yaml

RUN echo "conda activate OTPose" >> ~/.bashrc
ENV PATH /opt/conda/envs/OTPose/bin:$PATH
ENV CONDA_DEFAULT_ENV $OTPose
RUN apt install libc6 -y

RUN /bin/bash -c "source activate OTPose && conda install pytorch torchvision cudatoolkit=11.3 -c pytorch"
ENV CUDA_HOME "/usr/local/cuda"
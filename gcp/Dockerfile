# docker build --no-cache  multitasking .
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update \
        && apt-get install -y --no-install-recommends \
            git \
            ssh \
            htop \
            build-essential \
            locales \
            ca-certificates \
            curl \
            unzip \
            vim \
            binutils \
            libxext6 \
            libx11-6 \
            libglib2.0-0 \
            libxrender1 \
            libxtst6 \
            libxi6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda update --all
     # && \
#     /opt/conda/bin/conda install conda-build python=3.6.3 numpy pyyaml mkl
     # && \
#     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ADD repo-key /
RUN \
  chmod 600 /repo-key && \
  echo "IdentityFile /repo-key" >> /etc/ssh/ssh_config && \
  echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

RUN git clone git@github.com:MetaMind/rl-explanation.git && cd rl-explanation

#RUN conda install -c pytorch pytorch=1.0.0 cuda90 torchvision
RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext


# essentials
RUN pip install gym
RUN pip install pyyaml
RUN pip install gym
RUN pip install tqdm
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install tensorboardX
RUN pip install python-dateutil
RUN pip install ujson
RUN pip install wandb
RUN pip install Pillow
RUN pip install sklearn
RUN pip install pandas
RUN pip install gym_minigrid
RUN pip install tensorboard


CMD bash

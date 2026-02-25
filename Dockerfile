FROM scirocco2017/multireaderdevcontainer:1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libatlas-base-dev gfortran \
    python3-dev \
    rsync  \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

RUN set -euo pipefail \
  && cd /tmp \
  && git clone --branch 4.12.0 --depth 1 https://github.com/opencv/opencv.git \
  && git clone --branch 4.12.0 --depth 1 https://github.com/opencv/opencv_contrib.git \
  && mkdir -p /tmp/opencv/build \
  && cd /tmp/opencv/build \
  && cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/opt/opencv-4.12 \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=8.7 \
    -D WITH_GSTREAMER=ON \
    -D WITH_GTK=ON \
    -D WITH_GTK_3_X=ON \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
    .. \
  && make -j"$(nproc)" \
  && make install \
  && ldconfig \
  && rm -rf /tmp/opencv /tmp/opencv_contrib

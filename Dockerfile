FROM stablebaselines/stable-baselines3-cpu:1.1.0a5

RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    xvfb \
    libxrandr2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /root/code
ENV VENV /root/venv
COPY requirements.txt /tmp/


RUN \
    mkdir -p ${CODE_DIR}/rl_zoo && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install git+https://github.com/eleurent/highway-env && \
    rm -rf $HOME/.cache/pip

# tmp fix: current pypi gym package is out of date
# and create empty video files
RUN pip install gym==0.17.3

ENV PATH=$VENV/bin:$PATH

WORKDIR $CODE_DIR/rl_zoo

COPY ./ ./
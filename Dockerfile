FROM python:3.9

RUN apt-get -y update \
    && apt-get -y install swig

ENV CODE_DIR /root/code
ENV VENV /root/venv
COPY requirements.txt /tmp/

# # RUN pip install --upgrade pip

RUN \
    mkdir -p ${CODE_DIR}/rl_zoo && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install -U redis && \
    # pip install pip install highway-env==1.5.0 && \
    rm -rf $HOME/.cache/pip

# tmp fix: current pypi gym package is out of date
# and create empty video files
# RUN pip install gym==0.17.3

RUN pip install git+https://github.com/vinerich/zinc-coating-gym-env.git

ENV PATH=$VENV/bin:$PATH

WORKDIR /root/code
COPY ./ ./

RUN chmod +x ./run/hyperparameters_redis.sh
CMD ./suite.sh

RUN chmod +x ./suite.sh
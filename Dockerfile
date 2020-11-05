# pull official base image
FROM python:3.6-buster

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update -y && apt-get install -y cmake git
RUN git clone https://github.com/coccoc/coccoc-tokenizer.git
RUN pip install cython && cd coccoc-tokenizer && mkdir build && cd build \
    && cmake -DBUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX=~/.local .. \
    && make install \
    && cd ../python && python setup.py install \
    && cd / && rm -rf coccoc-tokenizer

# copy project
COPY . /usr/src/app

CMD ["/bin/bash"]
# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

#################### BASE BUILD IMAGE ####################
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /opt/vllm/

# install build and runtime dependencies
COPY src/ /opt/vllm

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/base.txt  
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/dev.txt


#################### EXTENSION BUILD IMAGE ####################
FROM dev AS build

# install build dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/build.txt

# cuda arch list used by torch
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

RUN python3 setup.py build_ext --inplace


#################### TEST IMAGE ####################
# image to run unit testing suite
FROM dev AS test

# copy pytorch extensions separately to avoid having to rebuild
# when python code changes
WORKDIR /usr/local/lib/python3.10/dist-packages
# ADD is used to preserve directory structure
ADD . /usr/local/lib/python3.10/dist-packages/
COPY --from=build /usr/local/lib/python3.10/dist-packages/vllm/*.so /usr/local/lib/python3.10/dist-packages/vllm/
# ignore build dependencies installation because we are using pre-complied extensions
RUN rm pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip VLLM_USE_PRECOMPILED=1 pip install . --verbose


#################### RUNTIME BASE IMAGE ####################
# use CUDA base as CUDA runtime dependencies are already installed via pip
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS base

# libnccl required for ray
RUN apt-get update -y \
    && apt-get install -y python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /tmp
COPY --from=build /opt/vllm/requirements/base.txt /tmp/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/base.txt  && \
    rm /tmp/base.txt

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM base AS sandbox
# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate

WORKDIR /usr/local/lib/python3.10/dist-packages/vllm/
COPY --from=build /opt/vllm/vllm/*.so /usr/local/lib/python3.10/dist-packages/vllm/
COPY src/vllm/ /usr/local/lib/python3.10/dist-packages/vllm/

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
# ENTRYPOINT ["python", "-m", "vllm.entrypoints.api_server"]

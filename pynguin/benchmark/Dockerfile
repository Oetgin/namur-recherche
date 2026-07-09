# SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
# SPDX-License-Identifier: MIT

################################################################################
# Dockerfile to run the benchmark for Pynguin.                                 #
#                                                                              #
# Build with `docker build -t pynguin-benchmark -f benchmark/Dockerfile .`     #
# from the root directory of the Pynguin repository.                           #
#                                                                              #
# The LLM provider port should be mapped to the host machine, e.g., using      #
# `-p 11434:11434` to map the port 11434 of the container to the host machine. #
#                                                                              #
# To register results, mount a volume to the container, e.g., using            #
# `-v /path/to/results:/pynguin/results`.                                      #
################################################################################

# The Python version to use
ARG PYTHON_VERSION=3.10.16

FROM python:${PYTHON_VERSION}-slim-bullseye AS pynguin-benchmark
ENV POETRY_VERSION="1.8.4"

WORKDIR /pynguin

# Python shall not write the byte code to *.pyc files; they cannot be cached between
# runs of the container anyway, hence we save the required time and resources for that
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering output that is written to STDOUT/STDERR; this allows to
# monitor the output in real time
ENV PYTHONUNBUFFERED=1

COPY . .

RUN pip install poetry==$POETRY_VERSION 
RUN poetry config virtualenvs.create false

RUN apt-get update
RUN apt-get install -y --no-install-recommends g++

RUN poetry install --with sutlibs --all-extras

# Set environment variables
# Set the Pynguin version
ENV PYNGUIN_VERSION="0.46.0.dev0"
# Pynguin requires to set the variable to show it that the user is aware that running
# Pynguin executes third-party code, which could cause arbitrary harm to the system.
# By setting the variable, the user acknowledges that they are aware of this.  In the
# container this is not too much of an issue (if we forget about breaking out of the
# container and similar things, of course).
ENV PYNGUIN_DANGER_AWARE=""
# Python shall not write the byte code to *.pyc files; they cannot be cached between
# runs of the container anyway, hence we save the required time and resources for that
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering output that is written to STDOUT/STDERR; this allows to
# monitor the output in real time
ENV PYTHONUNBUFFERED=1
# Disable random hash seeding for reproducible runs
ENV PYTHONHASHSEED=0

RUN apt-get install -y --no-install-recommends \
    git \
    python-tk
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "benchmark/config.py"]
CMD []

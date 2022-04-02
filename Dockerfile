# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN apt-get -y update
RUN apt-get -y install git

# RUN apt-get -y install ntp
WORKDIR /app
COPY . /app

RUN git config --global user.email "lutzurb1@students.zhaw.ch"
RUN git config --global user.name "Urban Lutz"
RUN git config --global http.sslVerify false

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python", "original_code/Synaptic-Flow/main.py"]

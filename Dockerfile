FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel


COPY requirements.txt .
RUN python -m pip install -r requirements.txt
WORKDIR /app
COPY . /app
CMD ["main.py"]
ENTRYPOINT [ "python" ]
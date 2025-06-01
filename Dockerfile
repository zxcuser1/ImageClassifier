
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip


RUN pip install --upgrade pip


WORKDIR /app
COPY . /app


RUN pip install -r requirements.txt


CMD ["python", "main.py"]

FROM python:3.9-slim-bullseye

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*
	
WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libx11-dev \
        libgtk-3-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir dlib==19.22.1

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]

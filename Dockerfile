FROM python:3.12-slim

# Set working directory
WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 9000


CMD ["python", "main.py"]

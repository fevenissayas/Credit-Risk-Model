FROM python:3.12-slim-bullseye

WORKDIR /home/feven/Desktop/Credit-Risk-Model

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --default-timeout=100

COPY . .

ENV MLFLOW_TRACKING_URI="file:///home/feven/Desktop/Credit-Risk-Model/mlruns"
ENV PYTHONPATH="/home/feven/Desktop/Credit-Risk-Model"

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
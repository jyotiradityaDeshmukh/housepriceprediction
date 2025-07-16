FROM python:latest AS base 

WORKDIR /app

COPY /src /app/src
COPY data/ /app/data
COPY requirements.txt .
COPY models/ /app/models

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=base /usr/local /usr/local
COPY /src /app/src
COPY data/ /app/data
COPY models/ /app/models

EXPOSE 5000

CMD ["python", "src/evaluate.py"]


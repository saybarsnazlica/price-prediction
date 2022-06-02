# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

WORKDIR /predict-price

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

ENV FLASK_APP=./api/app.py
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
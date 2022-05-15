FROM python:3.8

WORKDIR /myproject/venv

ADD main.py .

ADD model.pkl .

RUN pip install requests pandas flask

CMD ["python", "./main.py"]
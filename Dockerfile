FROM python:3.10

EXPOSE 80

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./app.py /code/app.py
COPY ./processed_data /code/processed_data
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "app:app","--host","0.0.0.0","--port","80"]
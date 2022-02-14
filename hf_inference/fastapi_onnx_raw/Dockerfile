FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY . /code/app
COPY ./mnli_onnx_model /code/mnli_onnx_model

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

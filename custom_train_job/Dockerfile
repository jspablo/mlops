FROM tensorflow/tensorflow:2.6.0-gpu

WORKDIR /train
COPY . /train

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3", "train.py" ]

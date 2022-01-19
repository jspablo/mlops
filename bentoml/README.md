# BentoML

Building a docker image for serving any model with a production-ready endpoint.

Create a python environment and set its dependencies
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

Create Bento service

```
python3 bento_service.py
```

Containerize the service

```
bentoml containerize TransformerZeroShotService:latest -t bentoml-zeroshot:latest
```

Run it and test it with Swagger UI

```
docker run -p 5000:5000 bentoml-zeroshot:latest --workers 2
```



![](/bentoml/screenshots/bentoml_swagger.gif)

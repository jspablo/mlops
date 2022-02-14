import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (BartForSequenceClassification,
    BartTokenizer, pipeline)


app = FastAPI()


class PredictionRequest(BaseModel):
    text: str
    categories: list
    multi_label: bool = True
    hypothesis_template: str = "The topic of this article is {}."


class PredictionResult(BaseModel):
    prediction: list


model_dir = os.environ.get("MODEL_DIR", "mnli_torch_model")
pipeline_name = "zero-shot-classification"
tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartForSequenceClassification.from_pretrained(model_dir)


@app.get("/predict", response_model=PredictionResult)
def read_root(prediction_request: PredictionRequest):
    classifier = pipeline(
        pipeline_name,
        model=model,
        tokenizer=tokenizer,
        framework="pt"
    )
    pred = classifier(
        prediction_request.text,
        prediction_request.categories,
        multi_label=prediction_request.multi_label,
        hypothesis_template=prediction_request.hypothesis_template
    )
    filtered_pred = [
        (label, prob)
        for label, prob in zip(
            pred.get('labels', []), pred.get('scores', [])
        )
        if prob > 0.5
    ]

    return {"prediction": filtered_pred}

import json
import os
from typing import Optional

import torch
from fastapi import FastAPI
from onnxruntime import InferenceSession
from pydantic import BaseModel
from transformers import (BartTokenizer, PreTrainedTokenizer,
                          ZeroShotClassificationPipeline)
from transformers.pipelines import ZeroShotClassificationArgumentHandler


app = FastAPI()


class PredictionRequest(BaseModel):
    text: str
    categories: list
    multi_label: bool = True
    hypothesis_template: str = "The topic of this article is {}."


class PredictionResult(BaseModel):
    prediction: list


class ONNXPipeline(ZeroShotClassificationPipeline):
    def __init__(
        self,
        model: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        entailment: int = -1,
        framework: str = "pt",
        num_workers: int = None,
        batch_size: int = None,
        args_parser = ZeroShotClassificationArgumentHandler(),
        device: int = -1,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.entailment = entailment
        self.framework = framework
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._args_parser = args_parser
        self.device = device if framework == "tf" \
            else torch.device("cpu" if device < 0 else f"cuda:{device}")
        self.call_count = 0

        (
            self._preprocess_params,
            self._forward_params,
            self._postprocess_params
        ) = self._sanitize_parameters(**kwargs)

        self.model_dir = model.replace("model.onnx", "")
        self.model = InferenceSession(model)
        self.label2id = label2id

    def _forward(self, inputs):
        candidate_label = inputs["candidate_label"]
        sequence = inputs["sequence"]

        outputs = self.model.run(
            output_names=["logits"],
            input_feed={
                "input_ids": inputs.get("input_ids").numpy(),
                "attention_mask": inputs.get("attention_mask").numpy()
            }
        )

        model_outputs = {
            "candidate_label": candidate_label,
            "sequence": sequence,
            "is_last": inputs["is_last"],
            "logits": torch.from_numpy(outputs[0])
        }

        return model_outputs

    @property
    def entailment_id(self):
        for label, ind in self.label2id.items():
            if label.lower().startswith("entail"):
                return ind
        return -1


model_name = os.environ.get("MODEL_NAME", "valhalla/distilbart-mnli-12-6")
model_file = os.environ.get("MODEL_FILE", "mnli_onnx_model/model.onnx") 
model_dir = os.environ.get("MODEL_DIR", "mnli_onnx_model")
tokenizer = BartTokenizer.from_pretrained(model_name)
with open(os.path.join(model_dir, "config.json")) as f:
    label2id = json.load(f).get("label2id")
classifier = ONNXPipeline(model=model_file, tokenizer=tokenizer, label2id=label2id)


@app.get("/predict", response_model=PredictionResult)
def read_root(prediction_request: PredictionRequest):    
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

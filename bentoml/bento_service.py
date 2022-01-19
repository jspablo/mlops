import bentoml
import torch
import transformers
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact
from transformers import pipeline, \
    BartForSequenceClassification, \
    BartTokenizer


@bentoml.env(pip_packages=["transformers==4.15.0", "torch==1.9.0"])
@bentoml.artifacts([TransformersModelArtifact("zeroshotmodel")])
class TransformerZeroShotService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        src_text = parsed_json.get("text", "test")
        pipeline_name = parsed_json.get("pipeline_name", "zero-shot-classification")
        categories = parsed_json.get("categories", ["test", "random"])

        print(transformers.__version__)
        print(torch.__version__)
        print(parsed_json)

        model = self.artifacts.zeroshotmodel.get("model")
        tokenizer = self.artifacts.zeroshotmodel.get("tokenizer")
        
        classifier = pipeline(
            pipeline_name,
            model=model,
            tokenizer=tokenizer,
            framework="pt"
        )
        pred = classifier(
            src_text,
            categories,
            multi_label=True,
            hypothesis_template="The topic of this article is {}."
        )
        filtered_pred = [
            (label, prob)
            for label, prob in zip(
                pred.get('labels', []), pred.get('scores', [])
            )
            if prob > 0.5
        ]

        return filtered_pred


if __name__ == "__main__":
    ts = TransformerZeroShotService()
    model_name = "valhalla/distilbart-mnli-12-6"
    model = BartForSequenceClassification.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    artifact = {"model": model, "tokenizer": tokenizer}
    ts.pack("zeroshotmodel", artifact)
    saved_path = ts.save()
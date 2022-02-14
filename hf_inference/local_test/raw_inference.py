import time

import torch
from transformers import AutoTokenizer, BartForSequenceClassification
from onnxruntime import InferenceSession


def raw_inference(inputs, model):
    t0 = time.time()
    logits = model(inputs)[0]
    print(f"Raw inference took {(time.time()-t0):.2f} sec")
    
    return logits


def quantization_inference(inputs, model):
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    t0 = time.time()
    logits = model(inputs)[0]
    print(f"Quantization inference took {(time.time()-t0):.2f} sec")
    
    return logits


def onnx_inference(inputs, session):
    t0 = time.time()
    logits = session.run(output_names=["logits"], input_feed=dict(inputs))
    print(f"ONNX inference took {(time.time()-t0):.2f} sec")

    return logits


if __name__ == "__main__":
    text = "Testing raw inference execution time!"
    model_name = "valhalla/distilbart-mnli-12-6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForSequenceClassification.from_pretrained(model_name)
    session = InferenceSession("onnx/distilbart/model.onnx")
    
    print("session_inputs", [(x.name, x.shape) for x in session.get_inputs()])
    print("session_outputs", [(x.name, x.shape) for x in session.get_outputs()])

    inputs = tokenizer(text, return_tensors="np")
    inputs_ids = tokenizer.encode(text, return_tensors="pt")

    raw_outputs = raw_inference(inputs_ids, model)
    quantization_outputs = quantization_inference(inputs_ids, model)
    onnx_outputs = onnx_inference(inputs, session)

    print("raw_outputs", raw_outputs)
    print("quantization_outputs", quantization_outputs)
    print("onnx_outputs", onnx_outputs)

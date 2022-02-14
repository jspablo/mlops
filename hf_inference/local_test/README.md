# Testing inference methods for Hugging Face Transformers

Exporting the model to the ONNX format

```python3 -m transformers.onnx --model=valhalla/distilbart-mnli-12-6 --feature=sequence-classification onnx/distilbart```

Compare the results for raw, quantization and ONNX

```python3 inferece_raw.py```

```
Raw inference took 0.06 sec

Quantization inference took 0.03 sec

ONNX inference took 0.04 sec

raw_outputs tensor([[-1.6986,  1.5381,  0.5525]], grad_fn=<AddmmBackward>)

quantization_outputs tensor([[-0.6780,  1.5914, -0.3257]])

onnx_outputs [array([[-1.6985686 ,  1.5381337 ,  0.55249757]], dtype=float32)]
```
from transformers import pipeline
from datasets import load_dataset
import sys


ds = load_dataset("cifar10", split="test[:1000]", cache_dir="dataset/train")

#0-999
i = 0
if len(sys.argv) > 1:
  i = int(sys.argv[1])

image = ds["img"][i]

classifier = pipeline("image-classification", model="model/checkpoint-2343")
result = classifier(image)

image.save("tmp.png")

print(result)
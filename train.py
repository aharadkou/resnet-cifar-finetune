from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, DefaultDataCollator, AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate
import numpy as np

dataset = "cifar10"
checkpoint = "microsoft/resnet-18"
training_args = TrainingArguments(
    output_dir="model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def get_img_transforms(image_processor):
  print("Image size: " + str(image_processor.size["shortest_edge"]))

  size = (
      image_processor.size["shortest_edge"]
      if "shortest_edge" in image_processor.size
      else (image_processor.size["height"], image_processor.size["width"])
  )

  _transforms = Compose([RandomResizedCrop(size), ToTensor(), Normalize(mean=image_processor.image_mean, std=image_processor.image_std)])

  def transforms(examples):
      examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["img"]]
      del examples["img"]
      return examples
  
  return transforms


def prepare_dataset(image_processor):
  transforms = get_img_transforms(image_processor)

  train = load_dataset(dataset, split="train", cache_dir="dataset/train")
  test = load_dataset(dataset, split="test", cache_dir="dataset/test")


  return train.with_transform(transforms), test.with_transform(transforms)

def prepare_labels(dataset_slice):
  labels = dataset_slice.features["label"].names

  label2id, id2label = dict(), dict()
  for i, label in enumerate(labels):
      label2id[label] = str(i)
      id2label[str(i)] = label
    
  return labels, label2id, id2label


def train():
  image_processor = AutoImageProcessor.from_pretrained(checkpoint)

  train, test = prepare_dataset(image_processor)

  labels, label2id, id2label = prepare_labels(train)

  data_collator = DefaultDataCollator()

  model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
  )

  trainer.train()

train()

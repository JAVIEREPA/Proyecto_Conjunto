from transformers import pipeline

pipe = pipeline("image-classification", model="joseluhf11/sign_language_classification_v1")

from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("joseluhf11/sign_language_classification_v1")
model = AutoModelForImageClassification.from_pretrained("joseluhf11/sign_language_classification_v1")
from PIL import Image

image = Image.open("imagenes/letra_A.jpg")

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)


logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

print("Clase predicha:", predicted_class_idx)

#predicted_class_idx = 10


predicted_letter = chr(predicted_class_idx + 65) 

print(f"Clase predicha: {predicted_class_idx}, que corresponde a la letra: {predicted_letter}")
from transformers import pipeline
from PIL import Image as PILImage

# Load the pipeline with the model and feature extractor
image_classifier = pipeline('image-classification',
                            model = './vit_fruit_cls',
                            feature_extractor = './vit_fruit_cls')

print("model loaded.....")

label = ['apple', 'asian pear', 'banana', 'cherry']
img_path = "test/apple/apple_403.png"
image = PILImage.open(img_path)
pred = image_classifier(image)
print(pred)

pred_label = pred[0]['label']
score = pred[0]['score']
pred_label_idx = int(pred_label.split('_')[-1])
fruit_name = label[pred_label_idx]
print(f"Predicted Fruit name: {fruit_name} with score {score}")


# img_path = "test/apple/apple_403.png"
# output:
# [{'label': 'LABEL_0', 'score': 0.9858248233795166},
#   {'label': 'LABEL_3', 'score': 0.00613699434325099},
#     {'label': 'LABEL_2', 'score': 0.004160773009061813},
#       {'label': 'LABEL_1', 'score': 0.003877444425597787}]
# Predicted Fruit name: apple with score 0.9858248233795166

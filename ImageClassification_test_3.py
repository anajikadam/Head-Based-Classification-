from transformers import pipeline
from PIL import Image as PILImage

# Load the pipeline with the model and feature extractor
image_classifier = pipeline('image-classification',
                            model = './vit_fruit_cls',
                            feature_extractor = './vit_fruit_cls')

print("model loaded.....")

label = ['apple', 'asian pear', 'banana', 'cherry']

img_path_1 = "data/test/apple/apple_414.png"
img_path_2 = "data/test/apple/apple_499.png"
img_path_3 = "data/test/asian pear/asian pear_415.png"
img_path_4 = "data/test/asian pear/asian pear_418.png"
img_path_5 = "data/test/banana/banana_416.png"
img_path_6 = "data/test/banana/banana_421.png"
img_path_7 = "data/test/cherry/cherry_410.png"
img_path_8 = "data/test/cherry/cherry_419.png"

image_paths = [img_path_1, img_path_2, img_path_3, img_path_4,
               img_path_5, img_path_6, img_path_7, img_path_8]
images = [PILImage.open(image_path) for image_path in image_paths]

for idx, img in enumerate(images):
    print(f"{idx+1}, img path: {image_paths[idx]}")
    pred = image_classifier(img)
    # print(pred)
    pred_label = pred[0]['label']
    score = pred[0]['score']
    pred_label_idx = int(pred_label.split('_')[-1])
    fruit_name = label[pred_label_idx]
    print(f"Predicted Fruit name: {fruit_name} with score {score}")
    print("==========="*10)
    print()


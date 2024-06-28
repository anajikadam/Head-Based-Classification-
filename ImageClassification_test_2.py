from transformers import pipeline
from datasets import load_dataset
from PIL import Image as PILImage
import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

dataset = load_dataset("./data")
# test_size=0.75: 400*0.75=100, test =100 and train 300 records
test_sampled = dataset['test'].train_test_split(test_size=0.75, seed=42)['train']
print(test_sampled)

# Load the pipeline with the model and feature extractor
image_classifier = pipeline('image-classification',
                            model = './vit_fruit_cls',
                            feature_extractor = './vit_fruit_cls')

print("model loaded.....")

import matplotlib.pyplot as plt
import random


true_labels = []
pred_labels = []
scores = []  # List to store probability values

for item in tqdm(dataset['test']):
    # Convert the PyTorch tensor to a PIL Image
    # t_img = item['image'].permute(2, 0, 1)
    t_img = item['image']
    image = TF.to_pil_image(t_img)

    # Use the pipeline to predict the class of each image
    pred = image_classifier(image)
    pred_label = pred[0]['label']

    # Extract score for the predicted label
    score = pred[0]['score']

    # Convert predicted label to the corresponding index
    pred_label_idx = int(pred_label.split('_')[-1])

    true_labels.append(item['label'])
    pred_labels.append(pred_label_idx)
    scores.append(score)  # Append score to the list

# Convert lists to numpy arrays for metric calculation
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

classes = ['apple', 'asian pear', 'banana', 'cherry']
# Define a function to display predictions with class names
def display_predictions(dataset, true_labels, pred_labels, class_names, scores, num_samples=10):

    fig, axs = plt.subplots(num_samples, 4, figsize=(12, 24))
    for i in range(num_samples):
        # Randomly select a sample from the dataset
        # index = random.randint(0, len(dataset) - 1)
        index = random.randint(0, len(dataset['test'])- 1)
        image_ = dataset['test']['image'][index]
        image = TF.to_pil_image(image_)
        true_label = true_labels[index]
        pred_label = pred_labels[index]
        class_name = class_names[pred_label]  # Get class name from class_names list
        score      = scores[index]
        print(index, true_label, pred_label, class_name, score)

        # Display the image
        axs[i, 0].imshow(image)  # Convert to HWC format
        axs[i, 0].axis('off')

        # Display the predicted label with class name
        axs[i, 1].text(0.5, 0.5, f'Predicted: {class_name}', fontsize=12, ha='center')
        axs[i, 1].axis('off')

        # Display the true label with class name
        axs[i, 2].text(0.5, 0.5, f'True: {class_names[true_label]}', fontsize=12, ha='center')
        axs[i, 2].axis('off')

        # Display the score
        axs[i, 3].text(0.5, 0.5, f'Score: {score:.2f}', fontsize=12, ha='center')
        axs[i, 3].axis('off')

    plt.tight_layout()
    # plt.show()
    fig.savefig('predicted.png')   # save the figure to file
    plt.close(fig)

# Assuming you have true_labels, pred_labels, probabilities, and class_names
display_predictions(dataset, true_labels, pred_labels, classes, scores)

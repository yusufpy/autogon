import streamlit as st
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
import random

# Define the data augmentation function
def apply_augmentation(image):
    angle = random.randint(-20, 20)
    augmented_image = image.rotate(angle)
    
    if random.random() > 0.5:
        augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    brightness_factor = random.uniform(0.85, 1.15)
    enhanced_image = ImageEnhance.Brightness(augmented_image).enhance(brightness_factor)
    
    contrast_factor = random.uniform(0.85, 1.15)
    enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast_factor)
    
    color_image = ImageEnhance.Color(enhanced_image).enhance(random.uniform(0.8, 1.2))
    
    scale_factor = random.uniform(0.8, 1.2)
    new_size = (int(color_image.width * scale_factor), int(color_image.height * scale_factor))
    new_image = color_image.resize(new_size, Image.NEAREST)
    
    return new_image

# Create augmented images
input_dir = "input/head-ct-hemorrhage/head_ct/head_ct/"
output_dir = "working/data-augmentation/"
image_num = 6

for files in os.listdir(input_dir):
    if files.endswith(".png"):
        input_path = os.path.join(input_dir, files)
        image = Image.open(input_path).convert("L")
                
        for i in range(image_num):
            augmented_image = apply_augmentation(image)
            output_file = f"{os.path.splitext(files)[0]}_{i}.png"
            output_path = os.path.join(output_dir, output_file)
            augmented_image.save(output_path)

# Load and process labels
labels = pd.read_csv('input/head-ct-hemorrhage/labels.csv')
labels.rename(columns={' hemorrhage': 'hemorrhage'}, inplace=True)
labels['id'] = labels['id'].astype(int)
labels['id'] = labels['id'].apply(lambda x: str('%03d' % x) + ".png")

# Generate augmented labels
expanded_labels = pd.DataFrame(columns=['id', 'hemorrhage'])
for index, row in labels.iterrows():
    img_id = row['id'].replace('.png', '')
    hemorrhage_label = row['hemorrhage']
    for i in range(6):
        new_img_id = f"{img_id}_{i}.png"
        new_row = {'id': new_img_id, 'hemorrhage': hemorrhage_label}
        expanded_labels = pd.concat([expanded_labels, pd.DataFrame(new_row, index=[0])], ignore_index=True)

# Split the data into train and test sets
train_label_df, test_label_df = train_test_split(expanded_labels, test_size=0.10, shuffle=True)
train_label_df.to_csv('./train_csv.csv', index=False, header=True)
test_label_df.to_csv('./test_csv.csv', index=False, header=True)

# Define custom dataset class
class CTDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(self.root_dir + img_id).convert("L")
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return img, y_label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# Create data loaders
batch_size = 10
num_workers = 4
trainset = CTDataset(root_dir='/kaggle/working/data-augmentation/', 
                     annotation_file='./train_csv.csv', 
                     transform=transform)
trainloader = DataLoader(trainset, 
                         batch_size=batch_size,
                         shuffle=True, 
                         num_workers=num_workers, 
                         drop_last=True)

testset = CTDataset(root_dir='/kaggle/working/data-augmentation/', 
                    annotation_file='./test_csv.csv', 
                    transform=transform)
testloader = DataLoader(testset, 
                        batch_size=batch_size,
                        shuffle=False, 
                        num_workers=num_workers)

# Define the neural network model
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 256, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(256, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(224-6)*(224-6), 64),
            nn.Linear(64, 64),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model, optimizer, and criterion
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Classifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()

# Streamlit app
st.title("CT Scan Hemorrhage Detection")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    #st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform data augmentation
    augmented_images = []
    for i in range(1):
        augmented_image = apply_augmentation(image)
        augmented_images.append(augmented_image)

    st.image(augmented_images, caption=["Augmented Image 1"], 
             use_column_width=True)

    # Process and classify images
    transformed_images = [transform(augmented_image) for augmented_image in augmented_images]
    tensor_images = torch.stack(transformed_images).to(device)
    outputs = model(tensor_images)
    predictions = torch.argmax(outputs, dim=1)

    st.write("Predictions:")
    for i, pred in enumerate(predictions):
        st.write(f"Augmented Image {i+1}: {'Hemorrhage' if pred == 1 else 'No Hemorrhage'}")

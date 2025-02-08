from Malaria_Model import Malaria_Model
from Load_data import load_data
import numpy as np
import pandas as pd

data_folder = r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\images'
label_file = r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\Train.csv'


array_images, array_image_ids = load_data(data_folder, label_file, False)
print("Data loaded,", len(array_images), "images found, and", len(array_image_ids), "image IDs found")
df = pd.read_csv(label_file)
labels = df['label']
coordinates = df[['x', 'y']]

print(np.array(array_images), np.array(array_image_ids), np.array(labels), np.array(coordinates))
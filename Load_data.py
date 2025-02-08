import os
import cv2
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_pages
def load_data(data_dir, label_file, pdf_output=False, images_per_pdf=500):
    images = []
    image_ids = []
    labels = []
    coordinates = []
    # Read label file with columns: Image_ID, class, confidence, ymin, xmin, ymax, xmax
    df = read_csv(label_file)
    # Group all rows by image
    grouped = df.groupby("Image_ID")
    
    # Initialize PDF variables if needed
    current_pdf = None
    pdf_idx = 1
    count_in_pdf = 0

    for image_id, group in grouped:
        print("Processing image:", image_id)
        image_path = os.path.join(data_dir, image_id)
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp']:
            print(f"Skipping non-image file: {image_path}")
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image: {image_path}")
            continue
        
        neg_present = False
        image_labels = []      # Store labels for current image
        image_coords = []      # Store coordinates for current image
        
        # Draw each bounding box with label and confidence
        for idx, row in group.iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            cell_class = row['class']
            confidence = row['confidence']
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=10)
            # Save label and coordinates regardless of type
            image_labels.append(cell_class)
            image_coords.append((xmin, ymin, xmax, ymax))
            
            if str(cell_class).lower() not in ["no label", "neg"]:
                text = f"{cell_class} ({confidence:.2f})"
                cv2.putText(image, text, (xmin, max(ymin - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4, cv2.LINE_AA)
            else:
                neg_present = True

        if neg_present:
            h, w = image.shape[:2]
            cv2.putText(image, "NEG", (w - 400, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)
        
        # Plot and optionally save to PDF
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Image: {image_id}")
        plt.axis('off')
        
        if pdf_output:
            # Open a new PDF file if needed
            if count_in_pdf == 0:
                current_pdf = pdf_pages.PdfPages(f"Labeled_Images_{pdf_idx}.pdf")
            current_pdf.savefig()
            count_in_pdf += 1
            # Close current PDF file and prepare for next if reached limit
            if count_in_pdf == images_per_pdf:
                current_pdf.close()
                pdf_idx += 1
                count_in_pdf = 0
        
        plt.close()
        
        images.append(image)
        image_ids.append(image_id)
        labels.append(image_labels)
        coordinates.append(image_coords)
    
    # If some pages remain open in the last PDF, close it.
    if pdf_output and current_pdf is not None and count_in_pdf > 0:
        current_pdf.close()
    
    return np.array(images), np.array(image_ids), np.array(labels), np.array(coordinates)


if __name__ == "__main__":
    # Update the paths accordingly
    data_dir = r'path\to\images'
    label_file = r'path\to\label\file.csv'
    images, image_ids = load_data(data_dir, label_file, True,500)
    print(f"Processed {len(images)} images with IDs: {set(image_ids)}")
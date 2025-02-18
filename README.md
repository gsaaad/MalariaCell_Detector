# MalariaCell_Detector

## Overview

MalariaCell_Detector is a machine learning project aimed at detecting malaria-infected cells from microscopic images. This tool leverages advanced image processing techniques and deep learning algorithms to accurately identify and classify cells as either infected or uninfected.

## Features

- **High Accuracy**: Utilizes state-of-the-art convolutional neural networks (CNN) for precise detection.
- **User-Friendly**: Easy-to-use interface for uploading and analyzing cell images.
- **Fast Processing**: Efficient algorithms ensure quick analysis of images.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gsaaad/MalariaCell_Detector.git
   ```
2. Navigate to the project directory:
   ```bash
   cd MalariaCell_Detector
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the detection script:
   ```bash
   python detect.py --image_path /path/to/your/image.png
   ```
2. The results will be displayed, indicating whether the cell is infected or not.

## Dataset

The model is trained on a publicly available dataset of microscopic images of blood cells. The dataset can be found [here](https://www.kaggle.com/datasets/rajsahu2004/lacuna-malaria-detection-dataset).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or suggestions, please open an issue or contact me directly.

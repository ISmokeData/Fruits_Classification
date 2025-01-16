# Fruits Classification Project    [Website](https://ismokedata-fruits-classification-home-ryevil.streamlit.app/)

## Overview
This project is a **Fruits Classification System** built using the **Fruit-360 dataset** from Kaggle. The system classifies images of fruits into predefined categories. It uses a custom Convolutional Neural Network (CNN) model built with PyTorch and is deployed as an interactive web application using Streamlit, enabling users to upload or capture fruit images in real-time for classification.

---

## Features
- Custom CNN model (**FruitNet**) with 3 convolutional layers, pooling, dropout, and fully connected layers.
- Real-time image classification via a user-friendly Streamlit web interface.
- Utilizes GPU for training and testing in Google Colab.
- Saves the trained model for deployment.

---

## Dataset
- **Dataset Name:** Fruit-360
- **Source:** Kaggle
- **Content:** Pre-labeled images of various fruits.

---

## Model Summary Table
| Layer            | Output Shape       | Parameters |
|------------------|--------------------|------------|
| Conv2d (conv1)   | [32, 96, 96]      | 896        |
| MaxPool2d (pool) | [32, 48, 48]      | 0          |
| Conv2d (conv2)   | [64, 48, 48]      | 18,496     |
| MaxPool2d (pool) | [64, 24, 24]      | 0          |
| Conv2d (conv3)   | [128, 24, 24]     | 73,856     |
| MaxPool2d (pool) | [128, 12, 12]     | 0          |
| Linear (fc1)     | [512]             | 9,437,184  |
| Dropout          | [512]             | 0          |
| Linear (fc2)     | [num_classes]     | 65,536     |
| **Total**        | **-**             | **9,596,968** |

---

## Training Details
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Device:** GPU (Google Colab)
- **Transformations:** Applied to images for data augmentation and normalization.
- **Dataloader:** Efficient data handling and batching.

---

## Deployment
- **Framework:** Streamlit
- **Features:**
  - Upload an image or click a real-time image for classification.
  - Displays the predicted fruit class.

---

## How to Run
### 1. Prerequisites
- Python 3.x
- PyTorch
- Streamlit
- Required Libraries: numpy, matplotlib, torchvision, etc.

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/fruits-classification.git
cd fruits-classification
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)
To retrain the model, run the Jupyter Notebook:
```bash
jupyter notebook train_model.ipynb
```

### 5. Run the Web Application
```bash
streamlit run app.py
```

### 6. Upload or Capture an Image
- Use the interface to upload an image or take a real-time image.
- View the predicted class.

---

## Results
- Achieved high accuracy on the test set.
- Real-time classification through the web interface.

---

## Folder Structure
```
fruits-classification/
├── data/
├── models/
│   └── fruitnet.pth
├── app.py
├── train_model.ipynb
├── requirements.txt
└── README.md
```

---

## Future Enhancements
- Expand dataset to include more fruit categories.
- Optimize the model for faster inference.
- Add multilingual support in the web application.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Author
- **Dhanraj Verma** - [GitHub Profile](https://github.com/ISmokeData)

Feel free to contribute to this project by creating pull requests or reporting issues!


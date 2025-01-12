import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn

# Load the model
model_path = "fruit3610_model.pth"
class_names = ['Apple 6',
 'Apple Braeburn 1',
 'Apple Crimson Snow 1',
 'Apple Golden 1',
 'Apple Golden 2',
 'Apple Golden 3',
 'Apple Granny Smith 1',
 'Apple Pink Lady 1',
 'Apple Red 1',
 'Apple Red 2',
 'Apple Red 3',
 'Apple Red Delicious 1',
 'Apple Red Yellow 1',
 'Apple Red Yellow 2',
 'Apple hit 1',
 'Apricot 1',
 'Avocado 1',
 'Avocado ripe 1',
 'Banana 1',
 'Banana Lady Finger 1',
 'Banana Red 1',
 'Beetroot 1',
 'Blueberry 1',
 'Cabbage white 1',
 'Cactus fruit 1',
 'Cantaloupe 1',
 'Cantaloupe 2',
 'Carambula 1',
 'Carrot 1',
 'Cauliflower 1',
 'Cherry 1',
 'Cherry 2',
 'Cherry Rainier 1',
 'Cherry Wax Black 1',
 'Cherry Wax Red 1',
 'Cherry Wax Yellow 1',
 'Chestnut 1',
 'Clementine 1',
 'Cocos 1',
 'Corn 1',
 'Corn Husk 1',
 'Cucumber 1',
 'Cucumber 3',
 'Cucumber Ripe 1',
 'Cucumber Ripe 2',
 'Dates 1',
 'Eggplant 1',
 'Eggplant long 1',
 'Fig 1',
 'Ginger Root 1',
 'Granadilla 1',
 'Grape Blue 1',
 'Grape Pink 1',
 'Grape White 1',
 'Grape White 2',
 'Grape White 3',
 'Grape White 4',
 'Grapefruit Pink 1',
 'Grapefruit White 1',
 'Guava 1',
 'Hazelnut 1',
 'Huckleberry 1',
 'Kaki 1',
 'Kiwi 1',
 'Kohlrabi 1',
 'Kumquats 1',
 'Lemon 1',
 'Lemon Meyer 1',
 'Limes 1',
 'Lychee 1',
 'Mandarine 1',
 'Mango 1',
 'Mango Red 1',
 'Mangostan 1',
 'Maracuja 1',
 'Melon Piel de Sapo 1',
 'Mulberry 1',
 'Nectarine 1',
 'Nectarine Flat 1',
 'Nut Forest 1',
 'Nut Pecan 1',
 'Onion Red 1',
 'Onion Red Peeled 1',
 'Onion White 1',
 'Orange 1',
 'Papaya 1',
 'Passion Fruit 1',
 'Peach 1',
 'Peach 2',
 'Peach Flat 1',
 'Pear 1',
 'Pear 2',
 'Pear 3',
 'Pear Abate 1',
 'Pear Forelle 1',
 'Pear Kaiser 1',
 'Pear Monster 1',
 'Pear Red 1',
 'Pear Stone 1',
 'Pear Williams 1',
 'Pepino 1',
 'Pepper Green 1',
 'Pepper Orange 1',
 'Pepper Red 1',
 'Pepper Yellow 1',
 'Physalis 1',
 'Physalis with Husk 1',
 'Pineapple 1',
 'Pineapple Mini 1',
 'Pitahaya Red 1',
 'Plum 1',
 'Plum 2',
 'Plum 3',
 'Pomegranate 1',
 'Pomelo Sweetie 1',
 'Potato Red 1',
 'Potato Red Washed 1',
 'Potato Sweet 1',
 'Potato White 1',
 'Quince 1',
 'Rambutan 1',
 'Raspberry 1',
 'Redcurrant 1',
 'Salak 1',
 'Strawberry 1',
 'Strawberry Wedge 1',
 'Tamarillo 1',
 'Tangelo 1',
 'Tomato 1',
 'Tomato 2',
 'Tomato 3',
 'Tomato 4',
 'Tomato Cherry Red 1',
 'Tomato Heart 1',
 'Tomato Maroon 1',
 'Tomato Yellow 1',
 'Tomato not Ripened 1',
 'Walnut 1',
 'Watermelon 1',
 'Zucchini 1',
 'Zucchini dark 1']# Replace with your actual class names


# Define the same model architecture
class FruitNet(nn.Module):
    def __init__(self, num_classes):
        super(FruitNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize and load the model
model = FruitNet(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define preprocessing for input image
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.title("Fruit Classification with PyTorch")

# Choose between camera and file upload
option = st.radio("Choose an option to input an image:", ["Use Camera", "Upload File"])

if option == "Use Camera":
    # Real-time camera access
    camera_active = st.checkbox("Enable Camera")

    if camera_active:
        # Start video capture
        cap = cv2.VideoCapture(0)
        st.write("Camera is now active. Press the 'Capture' button to take a photo.")

        captured_image = None

        if st.button("Capture"):
            # Read a frame from the camera
            ret, frame = cap.read()
            if ret:
                # Convert to RGB
                captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(captured_image, caption="Captured Image", use_container_width=True)
            else:
                st.write("Failed to capture image. Please try again.")

        # Release the camera after use
        cap.release()

        if captured_image is not None:
            # Convert captured image to PIL format
            image = Image.fromarray(captured_image)

            # Preprocess the image
            input_image = transform(image).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = model(input_image)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]

            # Display the prediction
            st.write(f"Predicted Class: **{predicted_class}**")

elif option == "Upload File":
    # File uploader for image
    uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        input_image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(input_image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]

        # Display the prediction
        st.write(f"Predicted Class: **{predicted_class}**")
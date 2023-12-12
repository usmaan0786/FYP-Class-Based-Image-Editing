import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization

# Load the model with the custom object scope
custom_objects = {'InstanceNormalization': InstanceNormalization}

# Define paths to your models
model_paths = {
    "Blue": './models/red-to-blue.h5',
    "Green": './models/black-to-green.h5',
    "Red": './models/blue-to-red.h5',
    "Black": './models/green-to-black.h5',
}

st.title("IMAGER - Image Editing Platform")
# Initial model selection
selected_model_name = st.selectbox("Select Model", list(model_paths.keys()), index=1)
selected_model_path = model_paths[selected_model_name]
model_AtoB = load_model(selected_model_path, custom_objects=custom_objects)

def main():
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    edit_btn = st.button("Edit Image")
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Display the uploaded image
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if edit_btn:
            # Send image to the deep learning model's API endpoint
            response = handle_file_upload(uploaded_file)

            # Display the output image
            with col2:
                st.image(response, caption="Output Image", channels='RGB', clamp=True, use_column_width=True)

def handle_file_upload(uploaded_file):
    # Prepare the image data
    image_data = BytesIO(uploaded_file.read())

    # Send the image data to the deep learning model's API endpoint
    output_image = generate_image(image_data)

    return output_image

def generate_image(img):
    try:
        # Read and preprocess the uploaded image
        img = Image.open(img)
        img = img.resize((512, 512))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Generate image using the selected model
        predictions = model_AtoB.predict(img_array)
        output_image = predictions[0]

        # Normalize pixel values to [0, 1]
        output_image_normalized = (output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image))

        # Convert the normalized output image array to a PIL Image
        output_pil_image = Image.fromarray((output_image_normalized * 255).astype(np.uint8))

        return np.array(output_pil_image)
    except Exception as e:
        st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()

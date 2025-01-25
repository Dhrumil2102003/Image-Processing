# import os
# import cv2
# import numpy as np
# import streamlit as st
# from model import deeplabv3_plus  # Import your model
# from PIL import Image
# from io import BytesIO
# import io

# # Set the image dimensions
# H, W = 256, 256

# # Function to load the model
# @st.cache_resource 
# def load_model():
#     model = deeplabv3_plus((H, W, 3))
#     model.load_weights("files/model.keras")  # Load your trained model weights
#     return model

# # Function to preprocess the image
# def preprocess_image(image):
#     image = cv2.resize(image, (W, H))
#     image = image / 255.0  # Normalize the image
#     return np.expand_dims(image, axis=0)  # Add batch dimension

# # Function to postprocess the model output
# def postprocess_output(output):
#     output = output.squeeze()  # Remove batch dimension
#     output = (output > 0.5).astype(np.uint8)  # Binarize the output
#     output = cv2.resize(output, (W, H))
#     return output * 255  # Scale to 0-255

# # Function to remove background
# def remove_background(model, image):
#     image = preprocess_image(image)
#     output = model.predict(image)  # Get model prediction
#     mask = postprocess_output(output)  # Postprocess the output

#     # Apply the mask to remove the background
#     result = cv2.bitwise_and(image[0], image[0], mask=mask)

#     return result

# # Streamlit app
# def main():
#     st.title("Background Removal App")
#     st.write("Upload an image to remove its background.")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Read the uploaded image
#         image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         h, w, _ = image.shape
        
#         # Load the model
#         model = load_model()

#         # Remove background
#         result = remove_background(model, image)
#         result = cv2.resize(result, (w, h))
        
#         # Display the original and result images
#         st.image(image, caption='Original Image', use_column_width=True)
#         st.image(result, caption='Background Removed', use_column_width=True)
       
        
        
#         result_image = Image.fromarray(result)
#         buffered = io.BytesIO()
#         result_image.save(buffered, format="PNG")
#         result_bytes = buffered.getvalue()
        
#         st.download_button(
#             label="Download image",
#             data=result_bytes,
#             file_name="remove_background.png",
#             mime="image/png")
    
#         # st.download_button(data = br, file_name = 'background_removed.png', label = 'Download')
        
#         # st.image(result, caption='Background Removed', use_column_width=True)

# if __name__ == "__main__":
#     main()






import os
import cv2
import numpy as np
import streamlit as st
from model import deeplabv3_plus  # Import your model
from PIL import Image
import io

# Set the image dimensions
H, W = 256, 256

# Function to load the model
@st.cache_resource 
def load_model():
    model = deeplabv3_plus((H, W, 3))
    model.load_weights("files/model.keras")  # Load your trained model weights
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (W, H))
    image = image / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to postprocess the model output
def postprocess_output(output):
    output = output.squeeze()  # Remove batch dimension
    output = (output > 0.5).astype(np.uint8)  # Binarize the output
    output = cv2.resize(output, (W, H))
    return output * 255  # Scale to 0-255

# Function to remove background
def remove_background(model, image):
    image = preprocess_image(image)
    output = model.predict(image)  # Get model prediction
    mask = postprocess_output(output)  # Postprocess the output

    # mask is single-channel
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # data types are compatible
    mask = mask.astype(np.uint8)
    image_uint8 = (image[0] * 255).astype(np.uint8)

    # Apply the mask to remove the background
    result = cv2.bitwise_and(image_uint8, image_uint8, mask=mask)

    return result

# Streamlit app
def main():
    st.title("Background Removal App")
    st.write("Upload an image to remove its background.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Load the model
        model = load_model()

        # Remove background
        result = remove_background(model, image)
        result = cv2.resize(result, (w, h))
        
        # Display the original and result images
        st.image(image, caption='Original Image', use_column_width=True)
        st.image(result, caption='Background Removed', use_column_width=True)
       
        # Prepare the result image for download
        result_image = Image.fromarray(result)
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        result_bytes = buffered.getvalue()
        
        st.download_button(
            label="Download image",
            data=result_bytes,
            file_name="remove_background.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()


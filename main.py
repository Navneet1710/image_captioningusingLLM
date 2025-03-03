import streamlit as st
import os
from tempfile import mkdtemp
from tools import ImageCaptionTool, ObjectDetectionTool

# Create instances of your tools directly
caption_tool = ImageCaptionTool()
detection_tool = ObjectDetectionTool()

st.title("Ask a question about your image")
st.header("Please upload an image")

file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # Display the uploaded image
    st.image(file, use_container_width=True)

    # Get user query
    user_question = st.text_input("What would you like to do?", label_visibility="visible")

    # Save the uploaded image to a temporary file
    temp_dir = mkdtemp()
    temp_file_path = os.path.join(temp_dir, "image.jpg")
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())

        if user_question:
            # Basic check: see if the user wants to "describe" or "detect"
            # You can customize these keywords or add more advanced checks
            lower_q = user_question.lower()
            if "object" in lower_q or "detect" in lower_q or "co-ordinates" in lower_q:
                st.info("Using Object Detection Tool...")
                result = detection_tool.run(temp_file_path)
            elif "describe" in lower_q or "caption" in lower_q or "tell" in lower_q:
                st.info("Using Image Captioning Tool...")
                result = caption_tool.run(temp_file_path)
            else:
                # If the user question doesn't match any known keywords,
                # just provide a default response or prompt them for clarity.
                st.warning("Could not determine which tool to use. Please use words like 'object', 'detect', 'describe', or 'caption'.")
                result = None

            # Display the result if we got one
            if result is not None:
                st.success(f"Answer:\n{result}")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

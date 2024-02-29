import rasterio  # Library for handling GeoTIFF files
import cv2  # Import OpenCV for resizing
import numpy as np  # Import NumPy for array operations
# import tensorflow as tf  # Importing the ML framework
import keras
from PIL import Image  # Library for image saving

# Load your ML model
model = keras.models.load_model("unet_model.h5")

# Prepare an image for prediction
file_path = "test1.tif"
with rasterio.open(file_path) as src:
        green_band = src.read(1)
        red_band = src.read(2)
        nir_band = src.read(3)
        swir_band = src.read(4)

        # Create a 3D array with NIR, Red, Green bands
        test_image = np.dstack(src.read([3, 2, 1]))

        # Normalize the image to 0-1 range
        test_image = test_image / np.max(test_image)

        test_image = cv2.resize(test_image, (256, 256))
        test_image = np.expand_dims(test_image, axis=0) # Add a new axis for the number of images

        # Make a prediction using the model
        output = model.predict(test_image)

        transform = src.transform
        profile = src.profile
        crs = src.crs

        # Create a new raster file with the predicted mask
        output_path = r"C:\Users\ASUS\Desktop\predicted_mask.tif"
        with rasterio.open(output_path, 'w', driver='GTiff', height=output.shape[0], width=output.shape[1], count=1, dtype=str(output.dtype), crs=crs, transform=transform) as dst:
            dst.write(output, 1)

        print(f"Predicted mask saved to: {output_path}")

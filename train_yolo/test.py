from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
yolo_model_path=r"/kaggle/working/working/best/weights/best.pt"
model = YOLO(yolo_model_path)  # Replace with your trained model's path

# Path to the image you want to test
image_path = '/kaggle/input/test-image/KakaoTalk_20240919_171834771_01.jpg'
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (640, 640))

# Save the resized image temporarily (optional, if you want to check the resizing)
# cv2.imwrite('resized_image.jpg', resized_image)

# Run inference on the resized image
results = model(resized_image)

# Get the image with bounding boxes drawn on it
annotated_image = results[0].plot()  # Get the first result and plot it

# Display the image with bounding boxes using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for plotting
plt.axis('off')  # Hide the axes
plt.show()

# Optionally, you can save the result image with bounding boxes to a file
output_path = 'path/to/save/output_image.jpg'
cv2.imwrite(output_path, annotated_image)
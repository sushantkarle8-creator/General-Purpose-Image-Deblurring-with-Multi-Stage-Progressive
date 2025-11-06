# Clone the repository
!git clone https://github.com/swz30/MPRNet.git
%cd MPRNet

# Install dependencies
!pip install torch torchvision numpy opencv-python pillow scikit-image tqdm


# Create the required folders for pretrained models, demo images, and results
!mkdir -p pretrained_models demo/degraded results


from google.colab import files

# Upload the pretrained model manually
print("Please upload your pretrained model (mprnet_deblurring.pth)")

# Upload file from user
uploaded_model = files.upload()

# Move and rename the uploaded model to the expected path
import os

# Create the necessary directory
!mkdir -p pretrained_models

# Loop through the uploaded files
for filename in uploaded_model.keys():
    # Rename uploaded model to expected name
    new_name = "mprnet_deblurring.pth"
    os.rename(filename, f'pretrained_models/{new_name}')
    print(f"Moved and renamed {filename} -> pretrained_models/{new_name}")


# Fix the path for the Deblurring task
!mkdir -p Deblurring/pretrained_models/

# Copy the pretrained model to the Deblurring directory
!cp pretrained_models/mprnet_deblurring.pth Deblurring/pretrained_models/model_deblurring.pth


# Upload images to be deblurred
uploaded_images = files.upload()

# Move uploaded images into the 'demo/degraded' folder
for filename in uploaded_images.keys():
    !mv "{filename}" demo/degraded/


# List contents of the 'demo/degraded' folder
print("\n--- Checking contents of demo/degraded directory ---")
!ls -l demo/degraded/


# Run the demo script for deblurring task
!python demo.py --task Deblurring --input_dir demo/degraded --result_dir results/


import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
import os

# List contents of results directory
print("\n--- Checking results directory ---")
!ls -lR results/

# Find and display the result image
result_image_path = None
task_subdir = 'results/Deblurring'

# Locate the result image
if os.path.exists(task_subdir):
    files_in_subdir = [f for f in os.listdir(task_subdir) if os.path.isfile(os.path.join(task_subdir, f)) and not f.startswith('.')]
    if files_in_subdir:
        result_image_path = os.path.join(task_subdir, files_in_subdir[0])
        print(f"\nFound result image in subdirectory: {result_image_path}")

# If result image not found in subdirectory, check directly in 'results/'
if result_image_path is None:
    files_in_results = [f for f in os.listdir('results') if os.path.isfile(os.path.join('results', f)) and not f.startswith('.')]
    if files_in_results:
        result_image_path = os.path.join('results', files_in_results[0])
        print(f"\nFound result image directly in results/: {result_image_path}")

# Display the deblurred result image
if result_image_path:
    print(f"\n--- Displaying deblurred result image ---")
    img_result = cv2.imread(result_image_path)
    if img_result is not None:
        cv2_imshow(img_result)

        # Display the original for comparison
        original_image_path = os.path.join('demo/degraded', list(uploaded_images.keys())[0])
        print(f"\n--- Displaying original image for comparison ---")
        img_original = cv2.imread(original_image_path)
        if img_original is not None:
            cv2_imshow(img_original)
        else:
            print("Could not load the original image for comparison.")
    else:
        print(f"Error: Could not load image from {result_image_path}")
else:
    print("\nCould not automatically locate the result image file. Please check the directory listing above.")


!pip install gradio --quiet


import os
print(os.getcwd())


# If you haven't cloned yet:
!git clone https://github.com/swz30/MPRNet.git

# Change directory into the repo
%cd MPRNet

# Verify files
!ls


import sys
sys.path.append(os.getcwd())  # Add current directory to Python path


from Deblurring.MPRNet import MPRNet
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model
model = MPRNet()

# Load pretrained weights
checkpoint_path = 'Deblurring/pretrained_models/model_deblurring.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("Model loaded successfully!")

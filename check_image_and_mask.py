# # 1.2.276.0.7230010.3.1.4.8323329.5793.1517875190.751053.dcm.jpeg*
# # 1.2.276.0.7230010.3.1.4.8323329.1599.1517875168.549263.dcm_mask.jpeg*   1.2.276.0.7230010.3.1.4.8323329.5793.1517875190.751053.dcm_mask.jpeg*

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import cv2
# from torch.utils.tensorboard import SummaryWriter
# import torch

# # 1.2.276.0.7230010.3.1.4.8323329.5793.1517875190.751053.dcm.jpeg
# # Load the image
# image_path = '/data/jliang12/jpang12/dataset/Pneumothorax_segmentation/train_jpeg/1.2.276.0.7230010.3.1.4.8323329.3643.1517875178.799075.dcm.jpeg'
# img = cv2.imread(image_path)

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)  # Convert to [1, C, H, W]

# # Create a SummaryWriter for TensorBoard logging
# writer = SummaryWriter("logs/images")

# # Add the image to TensorBoard
# writer.add_image("DCM Mask Image", img_tensor, 0)

# # Close the writer
# writer.close()

# print("Image logged to TensorBoard")

# # Instructions to start TensorBoard
# print("To view the image, run the following command in your terminal:")
# print(f"tensorboard --logdir=logs/images")

# import cv2
# import matplotlib.pyplot as plt

# # Load the image
# image_path = '/data/jliang12/jpang12/dataset/Pneumothorax_segmentation/train_jpeg/1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951.dcm_mask.jpeg'
# img = cv2.imread(image_path)

# # Convert BGR to RGB for correct color display with matplotlib
# output_path = '/home/mmaddur1/Segmentation/output_trainmask2_image1.jpg'
# cv2.imwrite(output_path, img)

# print(f"Image saved at {output_path}. You can open it manually to view.")

# # Define the path to your file
# file_path = "/home/mmaddur1/Segmentation/Pneumothorax-Segmentation-in-chest-X-rays/train/masks.txt"

# # Read the contents of the file
# with open(file_path, 'r') as file:
#     filenames = file.readlines()

# # Strip any extra newline characters and sort the filenames
# filenames = [filename.strip() for filename in filenames]
# filenames.sort()

# # Write the sorted filenames back to the same file
# with open(file_path, 'w') as file:
#     for filename in filenames:
#         file.write(f"{filename}\n")

# print("Filenames have been sorted and saved.")

import torch
import gc

# Delete all PyTorch objects
# for obj in gc.get_objects():
#     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#         del obj

# Clear CUDA cache
torch.cuda.empty_cache()

# Reset the GPU
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()

# Run garbage collection
gc.collect()

# Explicitly clear CUDA memory
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()

# Print memory usage to verify
print(torch.cuda.memory_summary())
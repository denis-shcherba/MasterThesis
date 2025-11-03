import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
import robotic as ry

C = ry.Config()
C.addFile("$RAI_PATH/scenarios/pandaSingle.g")

C.setJointState([.0, .0, .0, -2., 0. ,2., -0.5])
camview = ry.CameraView(C)
camview.setCamera(C.getFrame("cameraWrist"))


C.addFrame("box") \
    .setPosition([0, .4, .7]) \
    .setShape(ry.ST.ssBox, size=[.2, .1, .04, 0.005]) \
    .setColor([1, 0, 0]) \
    .setContact(1) \
    .setMass(.1) \
    .setAttributes({"friction": .01}) 

C.view(True)
rgb, depth_scaled = camview.computeImageAndDepth(C)

#depth_scaled = crop_or_rescale_img(depth, False, True )

depth_norm = (depth_scaled - depth_scaled.min()) / (depth_scaled.max() - depth_scaled.min()) 
depth_norm_255 = (depth_norm * 255).astype(np.uint8)

# duplicate to 3 channels
input_img = np.stack([depth_norm_255] * 3, axis=-1)

plt.imshow(input_img, cmap='gray')
plt.show()

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
patch_size = model.config.patch_size

inputs = processor(images=input_img, return_tensors="pt")
batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size

outputs = model(**inputs)
last_hidden_states = outputs[0]

patch_features_flat = last_hidden_states[:, 1:, :].squeeze(0)  # Shape: [num_patches_flat, 768]

features_np = patch_features_flat.detach().numpy() 

pca = PCA(n_components=3) 
pca.fit(features_np) 
features_pca = pca.transform(features_np) 

min_vals = features_pca.min(axis=0)
max_vals = features_pca.max(axis=0)
features_display = (features_pca - min_vals) / (max_vals - min_vals)

feature_map_rgb = features_display.reshape(
    (num_patches_height, num_patches_width, 3)
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original Image
axes[0].imshow(input_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Feature Map (RGB)
axes[1].imshow(feature_map_rgb)
axes[1].set_title(f'DINO Features (PCA to 3D RGB, {num_patches_height}x{num_patches_width})')
axes[1].axis('off')

plt.tight_layout()
plt.show()
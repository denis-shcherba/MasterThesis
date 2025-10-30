from torch import nn
from transformers.image_utils import ChannelDimension
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DaEncoder(nn.Module):
    """DepthAnything Encoder"""
    def __init__(self):
        super(DaEncoder, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf", use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")

    def forward(self, image):
        call_kwargs = {"do_rescale": False}
        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            data_format=ChannelDimension.FIRST,
            **call_kwargs,
        )
        outputs = self.model(**inputs)
        return outputs.predicted_depth
    
if __name__ == "__main__":
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    # visualize the prediction
    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = predicted_depth * 255 / predicted_depth.max()
    depth = depth.detach().cpu().numpy()
    depth = Image.fromarray(depth.astype("uint8"))
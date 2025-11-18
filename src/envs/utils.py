import numpy as np
import robotic as ry
import cmath 
from collections import deque
import random
import cv2

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


def sample_points(points, n_samples=4096) -> np.ndarray:
    if len(points) > n_samples:
        indices = np.random.choice(len(points), n_samples, replace=False)
        sampled_points = points[indices]
        
        return sampled_points    
    else:
        return points

def rescale_img(img, rescale_size: int = 96) -> np.ndarray:
    post_img = cv2.resize(img, (rescale_size, rescale_size), interpolation=cv2.INTER_LINEAR)

    return post_img

def crop_img(img, crop_size: int = 96) -> np.ndarray:
    pass

def choose_starting_point(point_list, metric=""):
    """
    Chooses a starting point from a list based on the metric.
    - If metric is "cost", selects the point with the lowest cost.
    - Otherwise, picks a random point.
    """
    if metric == "cost":
        if isinstance(point_list, dict):  # Expecting {point: cost, ...}
            return min(point_list, key=point_list.get)  # Get point with lowest cost
        else:
            raise ValueError("Expected a dictionary of {point: cost} for metric='cost'")
    
    else:
        return random.choice(point_list)  # Random choice if no metric
    

def point_in_box_filtering(points, box_params, ignore_planes=None):
    """
    Filters points that are within a given 3D bounding box, with an option to ignore certain planes.

    This function is useful for creating semi-infinite selection spaces or culling points against
    a box while excluding one or more of its boundary planes.

    Parameters:
    points (np.ndarray): An Nx3 array of 3D points to be filtered.
    box_params (tuple): A tuple containing the center and size of the box:
        - center (array-like): The (cx, cy, cz) coordinates of the box's center.
        - size (array-like): The (sx, sy, sz) dimensions of the box along the x, y, and z axes.
    ignore_planes (list or set, optional): A list of strings specifying which planes to ignore
        in the filtering logic. If None, all planes are considered.
        Valid values are: 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z'.
        For example, ignoring 'max_z' means there will be no upper bound check on the z-axis.

    Returns:
    np.ndarray: A filtered array containing only the points that are inside the specified region.
    """
    # If no planes are to be ignored, default to an empty set for easier processing.
    if ignore_planes is None:
        ignore_planes = set()
    else:
        # Convert list to a set for efficient 'in' checks.
        ignore_planes = set(ignore_planes)

    center, size = box_params
    center = np.array(center)
    size = np.array(size)

    # Calculate the minimum and maximum corner coordinates of the box.
    min_bound = center - size / 2
    max_bound = center + size / 2

    # Perform initial boundary checks for all points against all axes.
    # This creates two boolean arrays of shape (N, 3).
    lower_mask = points >= min_bound
    upper_mask = points <= max_bound

    # Create a mapping from the plane identifier string to the corresponding
    # boolean mask and the axis index (0 for x, 1 for y, 2 for z).
    plane_to_axis = {
        'min_x': (lower_mask, 0), 'max_x': (upper_mask, 0),
        'min_y': (lower_mask, 1), 'max_y': (upper_mask, 1),
        'min_z': (lower_mask, 2), 'max_z': (upper_mask, 2)
    }

    # Iterate through the planes to be ignored and disable their corresponding checks.
    # We do this by setting the entire column for that check to True, effectively
    # making the condition pass for all points.
    for plane in ignore_planes:
        if plane in plane_to_axis:
            mask, axis_index = plane_to_axis[plane]
            mask[:, axis_index] = True

    # Combine the lower and upper bound checks with a logical AND.
    combined_mask = lower_mask & upper_mask

    # A point is considered inside if all its relevant axis checks are True.
    # np.all along axis=1 checks if all three values (for x, y, z) are True for each point.
    final_mask = np.all(combined_mask, axis=1)

    return points[final_mask]


def cuboid_corners_to_size_com(corner_points):
    """
    Transforms cuboid corners to its size and COM

    Parameters:
    points (np.ndarray): 8x3 array of cuboid corner.
    """

    # Transform corner points to sizes in x, y, z
    z = np.linalg.norm(corner_points[0]-corner_points[1], 2)
    y = np.linalg.norm(corner_points[1]-corner_points[3], 2)
    x = np.linalg.norm(corner_points[0]-corner_points[4], 2)
    center=np.sum(corner_points, axis=0)/len(corner_points)

    return center, [x, y, z]

def sample_cuboid_edges(C, box_frame, yaw, sides_to_sample=[True, True, True, True], sides_rel=False, samples=10):
    """
    Samples points from the edges of a cuboid except for specified sides.
    
    Parameters:
    C (ry.Config): Configuration
    box_frame (string): Frame name of the box
    yaw (float): Yaw rotation of the cuboid in radians
    sides_to_sample (list): List of four booleans indicating which sides to sample:
                           [right, top, left, bottom] - True to sample that side
    samples (int): Total number of samples to be drawn
    
    Returns:
    List of 3D points on the selected cuboid edges
    """
    x_size = C.getFrame(box_frame).getSize()[0]
    y_size = C.getFrame(box_frame).getSize()[1]
    box_com = C.getFrame(box_frame).getPosition()
    
    if sides_rel:
        index = int((yaw + np.pi / 4) // (np.pi / 2)) % 4
        sides_to_sample = shift_list(sides_to_sample, index)

    # Check if we have at least one side to sample
    if not any(sides_to_sample):
        raise ValueError("At least one side must be selected for sampling")
    
    # Count how many sides we're sampling
    sides_count = sum(sides_to_sample)
    
    # Calculate samples per side (distribute evenly)
    samples_per_side = [0, 0, 0, 0]
    base_samples = samples // sides_count
    remainder = samples % sides_count
    
    side_index = 0
    for i in range(4):
        if sides_to_sample[i]:
            samples_per_side[i] = base_samples
            if side_index < remainder:
                samples_per_side[i] += 1
            side_index += 1
    
    edge_points = []
    
    # Sample the right side (x = x_size/2, y ranges from -y_size/2 to y_size/2)
    if sides_to_sample[0] and samples_per_side[0] > 0:
        y_values = np.linspace(-y_size/2, y_size/2, samples_per_side[0])
        for y in y_values:
            local_point = np.array([x_size/2, y, 0])
            rotated_point = rotate_point(local_point, yaw)
            edge_points.append(box_com + rotated_point)
    
    # Sample the top side (y = y_size/2, x ranges from x_size/2 to -x_size/2)
    if sides_to_sample[1] and samples_per_side[1] > 0:
        x_values = np.linspace(x_size/2, -x_size/2, samples_per_side[1])
        for x in x_values:
            local_point = np.array([x, y_size/2, 0])
            rotated_point = rotate_point(local_point, yaw)
            edge_points.append(box_com + rotated_point)
    
    # Sample the left side (x = -x_size/2, y ranges from y_size/2 to -y_size/2)
    if sides_to_sample[2] and samples_per_side[2] > 0:
        y_values = np.linspace(y_size/2, -y_size/2, samples_per_side[2])
        for y in y_values:
            local_point = np.array([-x_size/2, y, 0])
            rotated_point = rotate_point(local_point, yaw)
            edge_points.append(box_com + rotated_point)
    
    # Sample the bottom side (y = -y_size/2, x ranges from -x_size/2 to x_size/2)
    if sides_to_sample[3] and samples_per_side[3] > 0:
        x_values = np.linspace(-x_size/2, x_size/2, samples_per_side[3])
        for x in x_values:
            local_point = np.array([x, -y_size/2, 0])
            rotated_point = rotate_point(local_point, yaw)
            edge_points.append(box_com + rotated_point)
    
    return edge_points

def shift_list(lst, shift):
    d = deque(lst)
    d.rotate(-shift)  # Negative rotates left, positive rotates right
    return list(d)

def rotate_point(point, yaw):
    """
    Rotates a 2D point by the given yaw angle
    
    Parameters:
    point (np.array): Local coordinates [x, y, z]
    yaw (float): Rotation angle in radians
    
    Returns:
    np.array: Rotated coordinates
    """
    x, y, z = point
    rotated_x = x * np.cos(yaw) - y * np.sin(yaw)
    rotated_y = x * np.sin(yaw) + y * np.cos(yaw)
    return np.array([rotated_x, rotated_y, z])

def find_nearest_cuboid_edge_center(C, box_frame, yaw):
    """
    Given a cuboid and its yaw rotation, find the center point of the edge closest to 0 radiants observer
    """
    x_size = C.getFrame(box_frame).getSize()[0]
    y_size = C.getFrame(box_frame).getSize()[1]
    book_com = C.getFrame(box_frame).getPosition()

    # dir to push push dir waypoint from com
    directions = [
        (-x_size / 2, 0),
        (0, y_size / 2),
        (x_size / 2, 0),
        (0, -y_size / 2)
    ]

    index = int((yaw + np.pi / 4) // (np.pi / 2)) % 4
    delta = complex(*directions[index]) * cmath.exp(1j * yaw)

    return book_com+np.array([delta.real, delta.imag, 0])

def gram_schmidt_orthonormalize(vec: np.ndarray) -> np.ndarray:
    """
    Takes a 1D 6-element NumPy array representing the first two columns of a
    rotation matrix and returns a valid SO(3) rotation matrix.

    The process involves:
    1. Normalizing the first vector (first column).
    2. Making the second vector orthogonal to the first and normalizing it.
    3. Calculating the third vector as the cross product of the first two.

    Args:
        vec: A 1D NumPy array of shape (6,) representing the first two columns
             [r11, r21, r31, r12, r22, r32].

    Returns:
        A 3x3 NumPy array representing a valid SO(3) rotation matrix.
    """
    # Input validation
    if vec.shape != (6,):
        raise ValueError("Input vector must have a shape of (6,)")

    # 1. Extract the first two columns from the input vector
    v1 = vec[:3]
    v2 = vec[3:]

    # --- Start of the Gram-Schmidt Process ---

    # 2. First column (u1): Normalize the first vector.
    # We must handle the edge case where the vector is all zeros.
    norm_v1 = np.linalg.norm(v1)
    if norm_v1 < 1e-9:  # Use a small tolerance for floating point
        # If v1 is a zero vector, it cannot be normalized.
        # We can fall back to a default, like the x-axis.
        u1 = np.array([1.0, 0.0, 0.0])
    else:
        u1 = v1 / norm_v1

    # 3. Second column (u2): Make the second vector orthogonal to u1 and normalize.
    # Project v2 onto u1
    projection_v2_on_u1 = np.dot(u1, v2) * u1
    
    # Subtract the projection from v2 to get the orthogonal component
    v2_orthogonal = v2 - projection_v2_on_u1

    # Normalize the orthogonal component to get u2
    norm_v2_ortho = np.linalg.norm(v2_orthogonal)
    if norm_v2_ortho < 1e-9:
        # If v2 was collinear with v1, the orthogonal part is a zero vector.
        # We need to generate an arbitrary vector orthogonal to u1.
        # A robust way is to find the smallest component of u1 and swap it
        # with another component, negating one, to create a perpendicular vector.
        if abs(u1[0]) < abs(u1[1]):
            u2 = np.array([-u1[1], u1[0], 0])
        else:
            u2 = np.array([0, -u1[2], u1[1]])
        u2 /= np.linalg.norm(u2) # Normalize this new vector
    else:
        u2 = v2_orthogonal / norm_v2_ortho

    # 4. Third column (u3): Calculate using the cross product.
    # The cross product of two orthonormal vectors is guaranteed to be
    # orthogonal to both and have a length of 1.
    # This ensures a right-handed coordinate system for the matrix.
    u3 = np.cross(u1, u2)

    # 5. Assemble the final SO(3) rotation matrix by stacking the columns.
    rotation_matrix = np.column_stack((u1, u2, u3))

    return rotation_matrix

def convert_6d_rot_matrix_to_quaternion(rot_matrix):
    """
    Converts a 6D rotation representation to a quaternion.
    
    Parameters:
    rot_matrix (np.ndarray): A 6D rotation representation (first 3 columns of a rotation matrix).
    
    Returns:
    np.ndarray: A quaternion representing the rotation.
    """
    # Reconstruct the full rotation matrix
    r1 = rot_matrix[:, 0]
    r2 = rot_matrix[:, 1]
    r3 = np.cross(r1, r2)
    rot_full = np.column_stack((r1, r2, r3))
    
    # Convert rotation matrix to quaternion
    quat = ry.Quaternion().setMatrix(rot_full).asArr()
    
    return quat



@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image,
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)
    else:
        image = Image.fromarray(image)
    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections
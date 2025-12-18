import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        640, 480,
        rs.format.bgr8,
        30
    )

    pipeline.start(config)

    # Crop parameters
    crop_left = 0
    crop_right = 0   # pixels removed from the right

    try:
        print("RealSense RGB stream with X-crop. Press 'q' to exit.")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Sanity check for crop bounds
            width = color_image.shape[1]
            if crop_left + crop_right >= width:
                raise ValueError("Crop exceeds image width")

            # X-axis crop: [:, 50:-50]
            cropped_image = color_image[:, crop_left:width - crop_right]

            cv2.imshow("RealSense RGB (Cropped)", cropped_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Streaming stopped.")

if __name__ == "__main__":
    main()

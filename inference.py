import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image, ImageOps

# Load the ONNX model
model_path = "/path/to/your/model"
ort_session = ort.InferenceSession(model_path)
input_name = ort_session.get_inputs()[0].name


# Function to scale the image while preserving its aspect ratio
def scale_image_proportionally(image, target_width):
    w_percent = (target_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    return image.resize((target_width, h_size))


# Function to scale bounding box coordinates back to the original image dimensions


class PostProcessor:
    def __init__(self, input_width, input_height, img_width, img_height, confidence_thres, iou_thres):
        self.input_width = input_width
        self.input_height = input_height
        self.img_width = img_width
        self.img_height = img_height
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

    def postprocess(self, input_image, output):
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output))
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        keypoints_list = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            confidence = outputs[i][4]

            # If the maximum score is above the confidence threshold
            if confidence >= self.confidence_thres:
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                keypoints = outputs[i][5:]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the score and box coordinates to the respective lists
                scores.append(confidence)
                boxes.append([left, top, width, height])
                keypoints_list.append(keypoints)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and keypoints corresponding to the index
            box = boxes[i]
            score = scores[i]
            keypoints = keypoints_list[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, keypoints, class_id=0)

        # Return the modified input image
        return input_image

    def draw_detections(self, image, box, score, keypoints, class_id):
        # Convert box to integers
        box = [int(coord) for coord in box]

        # Draw bounding box
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

        # Map class IDs to class labels
        class_labels = ["Helmet", "No Helmet"]

        # Draw class label and confidence
        label = f"Class {class_labels[class_id]}, Score: {score:.2f}"
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw keypoints
        for j in range(0, len(keypoints), 3):
            if j + 2 < len(keypoints):  # Ensure there are enough elements in the list
                conf = keypoints[j]
                x = round(keypoints[j + 1] * self.img_width)
                y = round(keypoints[j + 2] * self.img_height)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)


# Example usage
image_path = "path/to/your/image"
image = Image.open(image_path).convert('RGB')
image = scale_image_proportionally(image, 640)
image_array = np.array(image)

# Run inference on the model
image_input = np.transpose(image_array, [2, 0, 1]).astype(np.float32)
image_input = np.expand_dims(image_input, axis=0)
image_input = image_input.astype(np.float32) / 255.0
result = ort_session.run(None, {input_name: image_input})
output = result[0]

# Perform post-processing using the reference code
postprocessor = PostProcessor(640, 640, 640, 640, 0.5, 0.4)
output_image = postprocessor.postprocess(image_array, output)

# Display the result
cv2.imshow("Image with Bounding Box", output_image)
cv2.waitKey(0)


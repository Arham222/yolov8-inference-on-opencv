
# YOLOv8 Inference with OpenCV
This repository demonstrates the deployment of YOLOv8 for inference on edge devices using OpenCV. The YOLO (You Only Look Once) model is renowned for its accuracy in real-time object detection.

## Requirements
- Python 3.10
- OpenCV
- NumPy
- onnxruntime
- Pillow (PIL)
YOLOv8 model weights (Refer to Ultralytics YOLOv8 repository for pretrained weights in onnx format)
## Usage
- Clone the repository:

```bash
#Copy code
git clone https://github.com/Arham222/YOLOv8-Inference-OpenCV.git
#cd YOLOv8-Inference-OpenCV
```
## Install dependencies:

```bash
#Copy code
pip install -r requirements.txt
```
- Download the YOLOv8 model weights from the Ultralytics YOLOv8 repository and place them in the project directory.

- Update the model_path variable in inference.py with the path to the YOLOv8 ONNX model.

## Run the inference script:

```bash
#Copy code
python inference.py
```
## Acknowledgements
- This project is based on the Ultralytics YOLOv8 repository. Special thanks to the Ultralytics team and the YOLOv8 community for their outstanding work on the YOLOv8 model.

## References
- YOLOv8: Ultralytics YOLOv8
- OpenCV: OpenCV Documentation
- onnxruntime: ONNX Runtime GitHub
- Feel free to explore, experiment, and contribute! If you have any questions or suggestions, please open an issue.

Happy coding! 🚀✨

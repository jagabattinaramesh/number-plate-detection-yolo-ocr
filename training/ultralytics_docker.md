# Training YOLOv8 Using Ultralytics Dockerfile

## 1️⃣ Clone the Ultralytics Repository
```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# Inside VM
# Build and Run the Docker Container
docker build -t ultralytics-yolo -f docker/Dockerfile .

docker run --gpus all --rm -it \
-v $(pwd):/usr/src/ultralytics/data \
ultralytics_yolo bash

# Run YOLO Training
python ultralytics/cfg/train.py \
    --model yolov8n.pt \
    --data /path/to/dataset.yaml \
    --epochs 200 \
    --imgsz 640 \
    --device 0

# Retrieve the Best Model
/runs/detect/train/weights/best.pt

# Copy Best Model to your local system:
scp -i your-key.pem username@your-gcp-vm:/path/to/best.pt ./models/best.pt

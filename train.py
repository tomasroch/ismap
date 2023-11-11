from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model

#MPS umožní využít maximální potenciál M1,M2,M3... chipů
results = model.train(data='config.yaml', epochs=200, device='mps')


from ultralytics import YOLO
import supervision as sv

BLUR_LEVEL = 200

model = YOLO('./runs/detect/train5/weights/best.pt')  # load a custom model
results = model.predict(source='images/val/erotic-pink-nude-7751268.jpeg', save=False, save_txt=False) # base use of model

### Blur detections part
detections = sv.Detections.from_ultralytics(results[0])
image = results[0].orig_img

blur_annotator = sv.BlurAnnotator(BLUR_LEVEL)

annotated_frame = blur_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
sv.plot_image(image=annotated_frame, size=(16, 16))









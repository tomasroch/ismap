import os
import cv2
from ultralytics import YOLO
import supervision as sv
from PIL import Image



BLUR_LEVEL = 200

model = YOLO('./runs/detect/google2/weights/best.pt')  # load a custom model

directory = './images/test'
for filename in os.listdir(directory):
    if(filename.startswith('.')): #skip hidden file
        continue
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        results = model.predict(source=f, save=False, save_txt=False)  # base use of model

        ### Blur detections part
        detections = sv.Detections.from_ultralytics(results[0])
        image = results[0].orig_img

        blur_annotator = sv.BlurAnnotator(BLUR_LEVEL)

        annotated_frame = blur_annotator.annotate(
            scene=image.copy(),
            detections=detections
        )
        #show image directily
        #sv.plot_image(image=annotated_frame, size=(16, 16))

        # save a image using extension
        im = Image.fromarray(annotated_frame)
        b, g, r = im.split()
        im = Image.merge("RGB", (r, g, b))
        #im.show()
        im.save(directory + "/processed/" + filename)


print('end file')





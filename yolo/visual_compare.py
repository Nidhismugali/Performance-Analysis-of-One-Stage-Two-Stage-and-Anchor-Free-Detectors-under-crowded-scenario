import cv2
import numpy as np
from ultralytics import YOLO
import os

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

# Project folder path
BASE_DIR = r"C:\PROJECTS\YOLOv9"

# Models (paths always inside the same folder)
model10_path = os.path.join(BASE_DIR, "YOLOv10", "best10.pt")
model11_path = os.path.join(BASE_DIR, "YOLOv11", "best11.pt")

# Change ONLY this filename (place your test image inside YOLOv9 folder)
test_image_name = "night.png"   # <<--- CHANGE THIS ONLY

# Full image path
test_image_path = os.path.join(BASE_DIR, test_image_name)

# --------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------

model10 = YOLO(model10_path)
model11 = YOLO(model11_path)

# --------------------------------------------------------
# RUN PREDICTIONS
# --------------------------------------------------------

res10 = model10.predict(test_image_path, save=False, imgsz=640)[0].plot()
res11 = model11.predict(test_image_path, save=False, imgsz=640)[0].plot()

# Convert RGB → BGR for OpenCV
img10 = cv2.cvtColor(res10, cv2.COLOR_RGB2BGR)
img11 = cv2.cvtColor(res11, cv2.COLOR_RGB2BGR)

# Combine images side-by-side
combined = np.hstack((img10, img11))

# --------------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------------

save_dir = os.path.join(BASE_DIR, "visual_results")
os.makedirs(save_dir, exist_ok=True)

cv2.imwrite(os.path.join(save_dir, "yolov10_result.jpg"), img10)
cv2.imwrite(os.path.join(save_dir, "yolov11_result.jpg"), img11)
cv2.imwrite(os.path.join(save_dir, "comparison.jpg"), combined)

print("\n✔ Saved visual results to:")
print(save_dir)

# --------------------------------------------------------
# SHOW RESULTS DIRECTLY IN WINDOW
# --------------------------------------------------------

cv2.imshow("YOLOv10 vs YOLOv11 Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

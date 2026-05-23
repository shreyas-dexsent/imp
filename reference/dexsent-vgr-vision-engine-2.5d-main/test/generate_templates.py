"""Tests for `test.generate_templates`."""

# import cv2
# import numpy as np
# from pathlib import Path

# OUT_DIR = Path("assets/templates")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # ---------- partA: white rectangle ----------
# imgA = np.zeros((120, 180, 3), dtype=np.uint8)
# cv2.rectangle(imgA, (30, 30), (150, 90), (255, 255, 255), -1)
# cv2.putText(imgA, "A", (70, 75),
#             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

# cv2.imwrite(str(OUT_DIR / "partA.png"), imgA)

# # ---------- partB: white circle ----------
# imgB = np.zeros((120, 180, 3), dtype=np.uint8)
# cv2.circle(imgB, (90, 60), 35, (255, 255, 255), -1)
# cv2.putText(imgB, "B", (80, 110),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# cv2.imwrite(str(OUT_DIR / "partB.png"), imgB)

# print("Templates generated:")
# print(" - assets/templates/partA.png")
# print(" - assets/templates/partB.png")

from pathlib import Path

import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    Path("assets/templates").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("assets/templates/my_face.png", frame)
    print("Saved face template")

import cv2
import numpy as np


target_image = cv2.imread('target.jpg', cv2.IMREAD_GRAYSCALE)
if target_image is None:
    raise ValueError("❌ Could not load target image. Check the file path!")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")
sift = cv2.SIFT_create()
kp_target, des_target = sift.detectAndCompute(target_image, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

print("✅ Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

    if des_frame is not None:
        matches = flann.knnMatch(des_target, des_frame, k=2)

        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        if len(good_matches) > 10:
            src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = target_image.shape
                pts = np.float32([[0, 0], [0, h - 1],
                                  [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)
                cv2.putText(frame, "Object Detected!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

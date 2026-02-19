import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


def create_identity_mask(h, w, left_eye, right_eye):
    """
    Creates smooth mask around eyes & nose bridge
    """

    mask = np.zeros((h, w), dtype=np.float32)

    # midpoint between eyes (important identity region)
    cx = int((left_eye[0] + right_eye[0]) / 2)
    cy = int((left_eye[1] + right_eye[1]) / 2)

    # gaussian falloff
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask[y, x] = np.exp(-(dist**2) / (2*(0.18*w)**2))

    mask = mask / mask.max()
    return mask


def align_face(image, bbox):
    xmin, ymin, width, height = bbox

    face = image[ymin:ymin+height, xmin:xmin+width]
    if face.size == 0:
        return None, None

    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0]
    h, w, _ = face.shape

    # eye landmarks
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]

    left = np.array([left_eye.x * w, left_eye.y * h])
    right = np.array([right_eye.x * w, right_eye.y * h])

    # rotation
    dy = right[1] - left[1]
    dx = right[0] - left[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(face, M, (w, h))

    aligned = cv2.resize(aligned, (160, 160))

    # create identity mask (resize to match)
    # center of aligned face (stable)
    cx, cy = 80, 80

    mask = np.zeros((160,160), dtype=np.float32)

    for y in range(160):
      for x in range(160):
         dist = np.sqrt((x-cx)**2 + (y-cy)**2)
         mask[y,x] = np.exp(-(dist**2)/(2*(28**2)))

    mask = mask / mask.max()
    mask = np.stack([mask,mask,mask], axis=0)


    return aligned, mask

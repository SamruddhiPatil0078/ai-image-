import torch
import torch.nn.functional as F
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1

device = "cuda" if torch.cuda.is_available() else "cpu"

facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Freeze model
for param in facenet_model.parameters():
    param.requires_grad = False


# Tensor embedding (used for cloaking optimization)
def extract_embedding_tensor(face_tensor):

    # FaceNet normalization (VERY IMPORTANT)
    face_tensor = (face_tensor - 0.5) / 0.5

    emb = facenet_model(face_tensor)
    emb = F.normalize(emb, p=2, dim=1)
    return emb



def get_facenet_model():
    return facenet_model


# CV2 embedding (used for distance verification)
def extract_embedding_facenet(face_img):

    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160,160))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet_model(tensor)
        emb = F.normalize(emb, p=2, dim=1)

    return emb.cpu().numpy()[0]

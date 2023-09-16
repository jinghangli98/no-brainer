# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np
import pdb
import time

cap = cv2.VideoCapture(0)
embedding_list = []
import numpy as np

# Load the NPZ file with allow_pickle=True
lookuptable = np.load('lookuptable.npz', allow_pickle=True)
table={}
for idx, item in enumerate(lookuptable):
    table[item]=lookuptable[item]

lookuptable=table

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2.T)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

device = torch.device('cpu')
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20,keep_all=True, device=device) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval()

start_time = time.time()
frame_count = 0
while True:
    
    _, frame = cap.read()
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #detect all faces in the frame
 
        # initializing resnet for face img to embeding conversion
    try:
        boxes, _ = mtcnn.detect(frame)
        for box in boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
            roi = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    
            face, prob = mtcnn(roi, return_prob=True) 
            emb = resnet(face) # passing cropped face into resnet model to get embedding matrix
            emb = np.float16(emb.detach())

            for idx, item in enumerate(lookuptable):
                metrics = np.squeeze(cosine_similarity(emb, lookuptable[item][0]))
                relationship = lookuptable[item][1]
                print(f'{item}: {metrics}')
                if metrics > 0.75:
                    cv2.putText(frame, f"{item}", (int(box[0]), int(box[1])-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"{relationship}", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except:
        pass

    cv2.imshow("Webcam Feed", frame)

cap.release()
cv2.destroyAllWindows()

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import glob
import json

images = glob.glob('./photos/*')
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
embedding_list = []
IDs = []
table = {}
for image in images:
    name = image.split('/')[-1].split('.')[0]
    IDs.append(name)
    frame = cv2.imread(image)
    face, prob = mtcnn(frame, return_prob=True) 
    emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
    emb = np.array(emb.detach())
    embedding_list.append(emb) # resulten embedding matrix is stored in a list
    table[name] = emb
    
# Specify the file path where you want to save the dictionary
np.savez('./lookuptable.npz', **table)

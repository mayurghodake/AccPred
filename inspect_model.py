import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import torch
import sys

try:
    with open('accident_model.pkl', 'rb') as f:
        data = pickle.load(f)
        print(f"Type: {type(data)}")
        print(f"Content: {data}")
except Exception as e:
    print(f"Error: {e}")

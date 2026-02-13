import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys

try:
    data = torch.load('accident_model.pkl', map_location='cpu')
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
    print("Load successful")
except Exception as e:
    print(f"Error: {e}")

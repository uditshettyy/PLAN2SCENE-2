import torch
import cv2
import numpy as np
from shapely.geometry import Polygon
import networkx as nx
import trimesh
import segmentation_models_pytorch as smp

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCV: {cv2.__version__}")
print(f"Shapely: OK")
print(f"NetworkX: {nx.__version__}")
print(f"Trimesh: {trimesh.__version__}")
print(f"SMP: {smp.__version__}")
print("\nAll dependencies installed successfully!")
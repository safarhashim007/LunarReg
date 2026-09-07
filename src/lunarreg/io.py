import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    if im is None or min(im.shape)<64: raise ValueError(f'Unreadable or too small image: {p}')
    if float(im.std())<1: raise ValueError(f'Insufficient image texture: {p}')
    return im


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def json_write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

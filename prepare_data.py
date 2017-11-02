import numpy as np
import os
import gzip
import h5py
import logging
from scipy import stats
from sklearn import metrics

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na
    
# Logging
def create_logging(log_dir, filemode):
    # Write out to file
    i1 = 0
    while os.path.isfile(os.path.join(log_dir, "%05d.log" % i1)):
        i1 += 1
    log_path = os.path.join(log_dir, "%05d.log" % i1)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)
                
    # Print to console   
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging
    
def eer(pred, gt):
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred, drop_intermediate=True)
    
    eps = 1E-6
    Points = [(0,0)]+zip(fpr, tpr)
    for i, point in enumerate(Points):
        if point[0]+eps >= 1-point[1]:
            break
    P1 = Points[i-1]; P2 = Points[i]
        
    #Interpolate between P1 and P2
    if abs(P2[0]-P1[0]) < eps:
        EER = P1[0]        
    else:        
        m = (P2[1]-P1[1]) / (P2[0]-P1[0])
        o = P1[1] - m * P1[0]
        EER = (1-o) / (1+m)  
    return EER

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc)*np.sqrt(2.0)
    return d_prime

# Load data    
def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        video_id_list = hf.get('video_id_list')
        x = np.array(x)
        y = np.array(y)
        video_id_list = list(video_id_list)
        
    return x, y, video_id_list

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.
    
def bool_to_float32(y):
    return np.float32(y)
    
def transform_data(x, y):
    x = uint8_to_float32(x)
    y = bool_to_float32(y)
    return x, y


"""
Summary:  Predict on the feature of a new audio. 
Author:   Qiuqiang Kong
Created:  2017.11.12
Modified: - 
"""
import os
import numpy as np
import argparse

from main import _attention
from hat import serializations


def predict_new(args):
    workspace = args.workspace
    
    # Load model. 
    md_path = os.path.join(workspace, "models", "main", args.model_name)
    md = serializations.load(md_path)
    
    # Simulate new data. 
    x_new = np.random.normal(size=(3, 10, 128))   # (n_clips, n_time, n_in)
    
    # Obtain final classification probability on an audio clip. 
    [y] = md.predict(x_new)     # (n_clips, n_out)
    print("y.shape: %s" % (y.shape,))
    
    # Obtain intermedial classification & attention value in the neural network.
    observe_nodes = [md.find_layer('cla').output_, 
                     md.find_layer('att').output_]
                     
    f_forward = md.get_observe_forward_func(observe_nodes)  # Forward function. 
    [cla, att] = md.run_function(f_forward, x_new, batch_size=None, tr_phase=0.)
    
    print("classification.shape: %s" % (cla.shape,))    # (n_clips, n_time, n_out)
    print("attention.shape: %s" % (att.shape,))    # (n_clips, n_time, n_out)
    

if __name__ == '__main__':
    # Arguments. 
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    
    predict_new(args)
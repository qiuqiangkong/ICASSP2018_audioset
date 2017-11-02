"""
Summary:  Audio Set classification for ICASSP 2018 paper
Author:   Qiuqiang Kong, Yong Xu
Created:  2017.11.02
Modified: - 
"""
import os
import numpy as np
import h5py
import sys
import argparse
import time
import logging
import cPickle
from sklearn import metrics
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

import prepare_data as pp_data
sys.path.append("/user/HS229/qk00006/my_code2015.5-/python/Hat")
from data_generator import RatioDataGenerator

from hat.models import Model
from hat.layers.core import *
from hat.callbacks import SaveModel, Validation, Callback
from hat.optimizers import Adam
from hat import serializations


# Evaluate stats
def eval(md, x, y, out_dir, out_probs_dir):
    pp_data.create_folder(out_dir)
    
    # Predict
    t1 = time.time()
    (n_clips, n_time, n_freq) = x.shape
    (x, y) = pp_data.transform_data(x, y)
    [prob] = md.predict(x)
    prob = prob.astype(np.float32)
    
    # Dump predicted probabilites for future average
    if out_probs_dir:
        pp_data.create_folder(out_probs_dir)
        out_prob_path = os.path.join(out_probs_dir, "prob_%d_iters.p" % md.iter_)
        cPickle.dump(prob, open(out_prob_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    
    # Compute and dump stats
    n_out = y.shape[1]
    stats = []
    t1 = time.time()
    for k in xrange(n_out):
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y[:, k], prob[:, k])
        avg_precision = metrics.average_precision_score(y[:, k], prob[:, k], average=None)
        (fpr, tpr, thresholds) = metrics.roc_curve(y[:, k], prob[:, k])
        auc = metrics.roc_auc_score(y[:, k], prob[:, k], average=None)
        eer = pp_data.eer(prob[:, k], y[:, k])
        skip = 1000
        dict = {'precisions': precisions[0::skip], 'recalls': recalls[0::skip], 'AP': avg_precision, 
                'fpr': fpr[0::skip], 'fnr': 1. - tpr[0::skip], 'auc': auc}
        
        stats.append(dict)
    logging.info("Callback time: %s" % (time.time() - t1,))
    
    dump_path = os.path.join(out_dir, "md%d_iters.p" % md.iter_)
    cPickle.dump(stats, open(dump_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    logging.info("mAP: %f" % np.mean([e['AP'] for e in stats]))

# Attention Lambda function
def _attention(inputs, **kwargs):
    [cla, att] = inputs
    _eps = 1e-7
    att = T.clip(att, _eps, 1. - _eps)
    normalized_att = att / T.sum(att, axis=1)[:, None, :]
    return T.sum(cla * normalized_att, axis=1)
    
    
# Train the model
def train(args):
    cpickle_dir = args.cpickle_dir
    workspace = args.workspace
    
    # Path of hdf5 data
    bal_train_hdf5_path = os.path.join(cpickle_dir, "bal_train.h5")
    unbal_train_hdf5_path = os.path.join(cpickle_dir, "unbal_train.h5")
    eval_hdf5_path = os.path.join(cpickle_dir, "eval.h5")
    
    # Load data
    t1 = time.time()
    (tr_x1, tr_y1, tr_id_list1) = pp_data.load_data(bal_train_hdf5_path)
    (tr_x2, tr_y2, tr_id_list2) = pp_data.load_data(unbal_train_hdf5_path)    
    tr_x = np.concatenate((tr_x1, tr_x2))
    tr_y = np.concatenate((tr_y1, tr_y2))
    tr_id_list = tr_id_list1 + tr_id_list2

    (te_x, te_y, te_id_list) = pp_data.load_data(eval_hdf5_path)
    logging.info("Loading data time: %s s" % (time.time() - t1))
    
    logging.info(tr_x1.shape, tr_x2.shape)
    logging.info("tr_x.shape: %s" % (tr_x.shape,))
    
    (_, n_time, n_freq) = tr_x.shape
    
    # Build model
    n_hid = 500
    n_out = tr_y.shape[1]
    
    lay_in = InputLayer(in_shape=(n_time, n_freq))
    a = Dense(n_out=n_hid, act='relu')(lay_in)
    a = Dropout(p_drop=0.2)(a)
    a = Dense(n_out=n_hid, act='relu')(a)
    a = Dropout(p_drop=0.2)(a)
    a = Dense(n_out=n_hid, act='relu')(a)
    a = Dropout(p_drop=0.2)(a)
    cla = Dense(n_out=n_out, act='sigmoid', name='cla')(a)
    att = Dense(n_out=n_out, act='softmax', name='att')(a)
    
    # Attention
    lay_out = Lambda(_attention)([cla, att])
    
    # Compile model
    md = Model(in_layers=[lay_in], out_layers=[lay_out])
    md.compile()
    md.summary(is_logging=True)
    
    # Save model every several iterations
    call_freq = 1000
    dump_fd = os.path.join(workspace, "models", pp_data.get_filename(__file__))
    pp_data.create_folder(dump_fd)
    save_model = SaveModel(dump_fd=dump_fd, call_freq=call_freq, type='iter', is_logging=True)
    
    # Callbacks function
    callbacks = [save_model]
    
    batch_size = 500
    tr_gen = RatioDataGenerator(batch_size=batch_size, type='train')
    
    # Optimization method
    optimizer = Adam(lr=args.lr)
    
    # Train
    stat_dir = os.path.join(workspace, "stats", pp_data.get_filename(__file__))
    pp_data.create_folder(stat_dir)
    prob_dir = os.path.join(workspace, "probs", pp_data.get_filename(__file__))
    pp_data.create_folder(prob_dir)
    
    tr_time = time.time()
    for (tr_batch_x, tr_batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        # Compute stats every several interations
        if md.iter_ % call_freq == 0:
            # Stats of evaluation dataset
            t1 = time.time()
            te_err = eval(md=md, x=te_x, y=te_y, 
                          out_dir=os.path.join(stat_dir, "test"), 
                          out_probs_dir=os.path.join(prob_dir, "test"))
            logging.info("Evaluate test time: %s" % (time.time() - t1,))
            
            # Stats of training dataset
            t1 = time.time()
            tr_bal_err = eval(md=md, x=tr_x1, y=tr_y1, 
                              out_dir=os.path.join(stat_dir, "train_bal"), 
                              out_probs_dir=None)
            logging.info("Evaluate tr_bal time: %s" % (time.time() - t1,))
            
        # Update params
        (tr_batch_x, tr_batch_y) = pp_data.transform_data(tr_batch_x, tr_batch_y)
        md.train_on_batch(batch_x=tr_batch_x, batch_y=tr_batch_y, 
                        loss_func='binary_crossentropy', 
                        optimizer=optimizer, 
                        callbacks=callbacks)
            
        # Stop training when maximum iteration achieves
        if md.iter_ == call_freq * 31:
            break


# Average predictions of different iterations and compute stats
def get_avg_stats(args, file_name, bgn_iter, fin_iter, interval_iter):
    eval_hdf5_path = os.path.join(args.cpickle_dir, "eval.h5")
    workspace = args.workspace
    
    # Load ground truth
    (te_x, te_y, te_id_list) = pp_data.load_data(eval_hdf5_path)
    y = te_y
    
    # Average prediction probabilities of several iterations
    prob_dir = os.path.join(workspace, "probs", file_name, "test")
    names = os.listdir(prob_dir)
    
    probs = []
    iters = xrange(bgn_iter, fin_iter, interval_iter)
    for iter in iters:
        pickle_path = os.path.join(prob_dir, "prob_%d_iters.p" % iter)
        prob = cPickle.load(open(pickle_path, 'rb'))
        probs.append(prob)
    avg_prob = np.mean(np.array(probs), axis=0)

    # Compute stats
    t1 = time.time()
    n_out = y.shape[1]
    stats = []
    for k in xrange(n_out):
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y[:, k], avg_prob[:, k])
        avg_precision = metrics.average_precision_score(y[:, k], avg_prob[:, k], average=None)
        (fpr, tpr, thresholds) = metrics.roc_curve(y[:, k], avg_prob[:, k])
        auc = metrics.roc_auc_score(y[:, k], avg_prob[:, k], average=None)
        eer = pp_data.eer(avg_prob[:, k], y[:, k])
        
        skip = 1000
        dict = {'precisions': precisions[0::skip], 'recalls': recalls[0::skip], 'AP': avg_precision, 
                'fpr': fpr[0::skip], 'fnr': 1. - tpr[0::skip], 'auc': auc}
        
        stats.append(dict)
    logging.info("Callback time: %s" % (time.time() - t1,))
    
    # Dump stats
    dump_path = os.path.join(workspace, "stats", pp_data.get_filename(__file__), "test", "avg_%d_%d_%d.p" % (bgn_iter, fin_iter, interval_iter))
    pp_data.create_folder(os.path.dirname(dump_path))
    cPickle.dump(stats, open(dump_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    # Write out to log
    logging.info("bgn_iter, fin_iter, interval_iter: %d, %d, %d" % (bgn_iter, fin_iter, interval_iter))
    logging.info("mAP: %f" % np.mean([e['AP'] for e in stats]))
    auc = np.mean([e['auc'] for e in stats])
    logging.info("auc: %f" % auc)
    logging.info("d_prime: %f" % pp_data.d_prime(auc))
    
    
# Main
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--cpickle_dir', type=str)
    parser_train.add_argument('--workspace', type=str)
    parser_train.add_argument('--lr', type=float, default=1e-3)
    
    parser_get_avg_stats = subparsers.add_parser('get_avg_stats')
    parser_get_avg_stats.add_argument('--cpickle_dir', type=str)
    parser_get_avg_stats.add_argument('--workspace')

    args = parser.parse_args()
    
    # Logs
    logs_dir = os.path.join(args.workspace, "logs", pp_data.get_filename(__file__))
    pp_data.create_folder(logs_dir)
    logging = pp_data.create_logging(logs_dir, filemode='w')
    logging.info(os.path.abspath(__file__))
    logging.info(sys.argv)
    
    if args.mode == "train":
        train(args)
    elif args.mode == 'get_avg_stats':
        file_name=pp_data.get_filename(__file__)
        bgn_iter, fin_iter, interval_iter = 20000, 30001, 1000
        get_avg_stats(args, file_name, bgn_iter, fin_iter, interval_iter)
    else:
        raise Exception("Error!")
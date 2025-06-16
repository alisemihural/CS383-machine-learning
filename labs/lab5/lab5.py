#!/usr/bin/env python3
# author: Ali Ural
# date: 12-05-2025
# description: LAB 5 - SVMs
import numpy as np

# hyper-params
C=1.0
epochs=15
rng_seed=0
train_frac=2/3
lambda_=1/C

# read and shuffle
data=np.loadtxt("spambase.data",delimiter=",")
np.random.seed(rng_seed)
np.random.shuffle(data)
X,y=data[:,:-1],data[:,-1].astype(int)

# split
n_train=int(np.ceil(train_frac*len(X)))
X_tr_raw,X_val_raw=X[:n_train],X[n_train:]
y_tr,y_val=y[:n_train],y[n_train:]

# z-score and bias
mu=X_tr_raw.mean(axis=0)
std=X_tr_raw.std(axis=0,ddof=0)
X_tr=((X_tr_raw-mu)/std)
X_val=((X_val_raw-mu)/std)
X_tr=np.hstack([X_tr,np.ones((X_tr.shape[0],1))])
X_val=np.hstack([X_val,np.ones((X_val.shape[0],1))])

# Gradient ascent
m,d=X_tr.shape
w=np.zeros(d)
y_tr_pm=2*y_tr-1
t=0
for epoch in range(epochs):
    for i in range(m):
        t+=1
        eta=1/(lambda_*t)
        x_i=X_tr[i]
        y_i=y_tr_pm[i]
        if y_i*np.dot(w,x_i)<1:
            w=(1-eta*lambda_)*w+eta*y_i*x_i
        else:
            w=(1-eta*lambda_)*w

def predict(X):
    return (X@w>=0).astype(int)

def stats(y_true,y_pred):
    tp=((y_true==1)&(y_pred==1)).sum()
    fp=((y_true==0)&(y_pred==1)).sum()
    fn=((y_true==1)&(y_pred==0)).sum()
    prec=tp/(tp+fp) if tp+fp else 0
    rec=tp/(tp+fn) if tp+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    acc=(y_true==y_pred).mean()
    return acc,prec,rec,f1

prior_spam=y_tr.mean()
prior_ham=1-prior_spam
print(f"Class priors –  spam: {prior_spam:.3f},  not‑spam: {prior_ham:.3f}")

for name,Xs,ys in [("Train",X_tr,y_tr),("Validation",X_val,y_val)]:
    acc,prec,rec,f1=stats(ys,predict(Xs))
    print(f"{name}:  Accuracy={acc*100:5.2f}%   Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
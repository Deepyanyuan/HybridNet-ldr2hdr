# HybridNet-ldr2hdr
HybridNet：ldr2hdr via CNNs, detail in paper 《HybridNet: Learing to Reconstruct HDR Image from a Single LDR Image via Deep HDR Hybrid Network》
original HDR dataset come from online: DML-HDR {http://dml.ece.ubc.ca/data/DML-HDR/}, Fairchild-HDR {http://rit-mcsl.org/fairchild//HDRPS/HDRthumbs.html}, and Funt-HDR {https://www2.cs.sfu.ca/~colour/data/}.

step 0: generate training pairs by generate_train_pairs.py

step 1: create network, in this paper, we create a multi-ouput and multi-branch CNN network, detail in network.py

step 2: train this network by HybridNet_train.py. Note that in this paper, we use three datasets, the first two had been training and the last dataset (Funt-HDR) had no training.

step 3: test this network by HybridNet_test.py.

step 4: predict the results if input any LDR image by HybridNet_predict.py

step 5: performance comparison by Matlab code

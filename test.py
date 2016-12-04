from subprocess import call
import os
from time import time

def check(weights_seq, weights_cuda):
    relativeTolerance = 1e-6

    for i in xrange(len(weights_seq)):
        relativeError = weights_cuda[i] - weights_seq[i]
        print "Comparing, cuda:", weights_cuda[i], "seq:", weights_seq[i], "Diff:", relativeError
        if (relativeError > relativeTolerance) or (relativeError < -relativeTolerance):
            print "Failed."
            return
    print "Passed."

if __name__ == "__main__":
    
    weights_cuda = list();
    weights_seq = list();

    print "\n\n\n\n*************GPU*************\n\n"

    os.chdir("cuda")

    # t1 = time()

    call(["./run"], shell=True)

    # cuda_time = time() - t1

    with open("out.txt", "r") as f:
        for line in f.readlines()[1:]:
            weights_cuda.append(float(line.split(",")[2]))

    print "\n\n\n\n*************SEQUENTIAL*************\n\n"

    os.chdir("../sequential")

    # t1 = time()

    call(["./run"], shell=True)

    # seq_time = time() - t1

    with open("out.txt", "r") as f:
        for line in f.readlines()[1:]:
            weights_seq.append(float(line.split(",")[2]))

    print "cuda", weights_cuda#, "time: ", cuda_time
    print "seq", weights_seq#, "time:", seq_time

    
    check(weights_seq, weights_cuda)
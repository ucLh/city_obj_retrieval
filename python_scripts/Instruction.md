### Short instrucion

To obtain embedding network checkpoint you should use 
`train_softmax.py` script and then convert it to .pb format 
with `freeze_graph.py`. 

After that you can also use `optimize_pb.py` for checkpoint optimization.
It will remove redundant `"phase_train"` placeholder.
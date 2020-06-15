"""Strips phase_train node from a trained net's graph
"""
import argparse
import copy
import sys

import tensorflow as tf
from tensorflow.core.framework import graph_pb2


# load our graph
def load_graph(filename):
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def main(args):
    c = tf.constant(False, dtype=bool, shape=[], name='phase_train')

    graph_def = load_graph(args.input_graph)

    # Create new graph, and rebuild it from original one
    # replacing phase train node def with constant
    new_graph_def = graph_pb2.GraphDef()
    for node in graph_def.node:
        if node.name == 'phase_train':
            new_graph_def.node.extend([c.op.node_def])
        else:
            new_graph_def.node.extend([copy.deepcopy(node)])

    # save new graph
    with tf.gfile.GFile(args.output_graph, "wb") as f:
        f.write(new_graph_def.SerializeToString())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_graph', type=str,
                        help='Filename for the imported graphdef protobuf (.pb)')
    parser.add_argument('output_graph', type=str,
                        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

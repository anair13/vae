import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import digits_model

conf = digits_model.get_default_conf(run=1)

model = digits_model.DigitsModel(conf)

N = 10000
model.train(N, True)

# tf.reset_default_graph()

BATCHES = 50
for i in range(1000, N + 1, 1000):
    print i
    print "val", model.evaluate(i, BATCHES, "val")
    print "train", model.evaluate(i, BATCHES, "train")

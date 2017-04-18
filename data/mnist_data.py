from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

SOURCES = {"train": mnist.train, "val": mnist.validation, "test": mnist.test}
current = {"train": 0, "val": 0, "test": 0}

def get_batch(source_name, batch_size = 1):
    return SOURCES[source_name].next_batch(batch_size)
    # mnist.train.next_batch(100)
    # """X is the input to the network, Y is the supervision/ground truth"""
    # dataset = SOURCES[source_name]
    # i = current[source_name]
    # X = dataset.images[i:i+batch_size, :, :]
    # Y = dataset.labels[i:i+batch_size, :]
    # print X.shape, Y.shape

    # current[source_name] = (i + batch_size) % dataset.images.shape[0]

    # return X, Y

if __name__ == "__main__":
    print mnist
    print type(mnist.train.images)
    print mnist.train.images.shape
    print mnist.validation.images.shape
    print mnist.test.images.shape
    for i in range(5):
        X, Y = get_batch("train", 10)
        print Y.shape
        print X.shape

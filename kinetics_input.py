from dataset.kinetics_clustered import KineticsClustered
import tensorpack.dataflow as df
import cv2
import numpy as np
import tensorflow as tf

def dataflow2(num_reference=3, shuffle=True):
    kinetics_path = "/media/engin/63c43c7a-cb63-4c43-b70c-f3cb4d68762a/datasets/kinetics/kinetics700"
    ds = KineticsClustered(kinetics_path, num_frames=num_reference + 1,
                           skips=[0, 4, 4, 4][:num_reference + 1], shuffle=False)

    # ds = df.MapDataComponent(ds, lambda images: [cv2.resize(image, (256, 256)) for image in images], index=1)

    ds = df.MapData(ds, lambda dp: [dp[1][:num_reference], dp[2][:num_reference], dp[1][num_reference:], dp[2][num_reference:]])

    # # for images (ref, target)
    # for idx in [0, 2]:
    #     ds = df.MapDataComponent(ds, lambda images: [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(256, 256, 1) for image in images], index=idx)

    # stack for tensor
    ds = df.MapData(ds, lambda dp: [np.stack(dp[0] + dp[2], axis=0), np.stack(dp[1] + dp[3], axis=0)])

    ds = df.MapData(ds, tuple)  # for tensorflow.data.dataset
    # ds = df.MultiProcessPrefetchData(ds, nr_prefetch=256, nr_proc=num_process)
    # ds = df.PrefetchtraDataZMQ(ds, nr_proc=1)
    # ds = df.RepeatedData(ds, total_num_epoch)
    # ds = df.RepeatedData(ds, -1)
    return ds

def dataflow3(num_reference=3, num_sets = 1, shuffle=True):
    kinetics_path = "/media/engin/63c43c7a-cb63-4c43-b70c-f3cb4d68762a/datasets/kinetics/kinetics700"

    ds_list = []
    for i in range(num_sets):
        ds1 = KineticsClustered(i, kinetics_path, num_frames=num_reference + 1,
                               skips=[0, 4, 4, 4][:num_reference + 1], shuffle=False)
        ds1 = df.RepeatedData(ds1, -1)
        ds_list.append(ds1)

    # ds2 = KineticsClustered(1, kinetics_path, num_frames=num_reference + 1,
    #                        skips=[0, 4, 4, 4][:num_reference + 1], shuffle=False)
    # ds2 = df.RepeatedData(ds2, -1)

    ds = df.JoinData(ds_list)

    # ds = df.MapData(ds, lambda dp: [ [dp[0], dp[1], dp[2]] ])
    ds = df.MapData(ds, lambda dp: [[dp[i], dp[i + 1], dp[i + 2]] for i in range(0, num_sets*3, 3)])

    # for idx in [0, 1]:
    #     ds = df.MapDataComponent(ds, lambda dp: [dp[1][:num_reference], dp[2][:num_reference], dp[1][num_reference:], dp[2][num_reference:]], index=idx)


    # # stack for tensor
    for idx in range(num_sets):
        ds = df.MapDataComponent(ds, lambda dp: [np.stack(dp[1], axis=0), np.stack(dp[2], axis=0)], index=idx)

    ds = df.MapData(ds, tuple)  # for tensorflow.data.dataset

    # ds = df.BatchData(ds, 2, use_list=True)

    #Prepare epochs
    # ds = df.RepeatedData(ds, total_num_epoch)
    # ds = df.RepeatedData(ds, -1)
    return ds

def dataflow4(num_reference=3, order = 1, shuffle=True):
    kinetics_path = "/media/engin/63c43c7a-cb63-4c43-b70c-f3cb4d68762a/datasets/kinetics/kinetics700"

    ds = KineticsClustered(order, kinetics_path, num_frames=num_reference + 1,
                           skips=[0, 4, 4, 4][:num_reference + 1], shuffle=False)

    # # stack for tensor
    ds = df.MapData(ds, lambda dp: [np.stack(dp[1], axis=0), np.stack(dp[2], axis=0)])

    ds = df.MapData(ds, tuple)  # for tensorflow.data.dataset

    ds = df.RepeatedData(ds, -1)
    return ds

def get_dataflow(batch_size, num_sets):

    images_list = []
    labels_list = []
    for i in range(num_sets):
        ds = dataflow4(order=i)
        ds.reset_state()

        dataset = tf.data.Dataset.from_generator(ds.get_data,
                                       output_types=(tf.float32, tf.int64),
                                       output_shapes=(tf.TensorShape([4, 256, 256, 1]), tf.TensorShape([4, 32, 32, 1]))).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        images_list.append(images)
        labels_list.append(labels)

    return images_list, labels_list

    # ds = dataflow4(order=0)
    # ds.reset_state()
    #
    # dataset = tf.data.Dataset.from_generator(ds.get_data,
    #                                          output_types=(tf.float32, tf.int64),
    #                                          output_shapes=(
    #                                          tf.TensorShape([4, 256, 256, 1]), tf.TensorShape([4, 32, 32, 1]))).batch(batch_size)
    # iterator = dataset.make_one_shot_iterator()
    # images, labels = iterator.get_next()
    # return images, labels

if __name__ == "__main__":
    ds = dataflow4()
    a = 0
    for data in ds:
        a = a + 1

    b = 0
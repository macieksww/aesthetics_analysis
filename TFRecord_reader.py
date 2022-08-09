import tensorflow as tf 
def parser(data_record):
    ''' TFRecord parser '''

    feature_dict = {
      'label' : tf.io.FixedLenFeature([], tf.int64),
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width' : tf.io.FixedLenFeature([], tf.int64),
      'chans' : tf.io.FixedLenFeature([], tf.int64),
      'image' : tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(data_record, feature_dict)
    label = tf.cast(sample['label'], tf.int32)

    h = tf.cast(sample['height'], tf.int32)
    w = tf.cast(sample['width'], tf.int32)
    c = tf.cast(sample['chans'], tf.int32)
    image = tf.io.decode_image(sample['image'], channels=3)
    image = tf.reshape(image,[h,w,3])

    return image, label


def input_fn(tfrec_dir, batchsize):
    '''
    Dataset creation and augmentation pipeline
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/*.tfrecord'.format(tfrec_dir), shuffle=False)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(...add preprocessing in here.....)
    
    dataset = dataset.repeat()
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


train_dataset = input_fn(tfrec_dir, batchsize)
train_history = model.fit(train_dataset,
                        ..
                        )
                        
scores = model.evaluate(test_dataset,
                        ..
                        )
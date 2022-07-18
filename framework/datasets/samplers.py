import tensorflow as tf

# samplers for the datasets
# ==============================================================================

# class to map class_specific datasets to m samples per class datasets
class DatasetToRandomSamplesPerClass(object):
    def __init__(self, num_classes, sample_per_class, dataset_size, shuffle_classes=True):

        self.num_classes = num_classes
        self.sample_per_class = sample_per_class
        avg_sample_per_class = dataset_size / num_classes
        pick_repeats = avg_sample_per_class // sample_per_class
        self.pick_repeats = \
            int(pick_repeats if (pick_repeats == (avg_sample_per_class / sample_per_class))
                else pick_repeats + 1)


        self.shuffle_classes = shuffle_classes

    @tf.function(input_signature=[tf.RaggedTensorSpec(shape=(None, None), dtype=tf.dtypes.string, ragged_rank=1)])
    def __call__(self, class_specific_datasets):
        ''' input: datasets: is a RaggedTensor rows of which are class specific datasets
            returns: datasets: tf.data.Dataset whose elements are M samples from a particular class
             i. e. one can perform datasets.batch(N) to gather M samples each for N exclusive classes'''

        # use indices to shuffle and pick samples
        class_samples_indcs = tf.cast(tf.ragged.range(0, class_specific_datasets.row_lengths(axis=1)), dtype=tf.int32)

        # convert datasets to tensor so that batch gather can be used for efficiency
        class_specific_datasets_as_tensor = class_specific_datasets.to_tensor()

        total_picks_per_class = self.pick_repeats * self.sample_per_class

        # the body of the for loop which passes over each class specific datasets and performs sampling
        def shuffle_and_pick_m(k, picked_indcs_per_class):

            samples_indcs = class_samples_indcs[k]
            samples_queue_0 = tf.random.shuffle(samples_indcs)
            queue_ptr_0 = tf.constant(0)
            queue_capacity = tf.shape(samples_queue_0)[0]
            num_samples_in_queue_0 = queue_capacity
            p_0 = tf.constant(0)
            picked_indcs_0 = tf.TensorArray(dtype=tf.int32, size=total_picks_per_class, dynamic_size=False,
                                            clear_after_read=False, element_shape=tf.TensorShape([]))

            # the body of the inner for loop which picks <total_picks_per_class> samples randomly
            def pick_from_shuffle_queue(p, picked_indcs, samples_queue, queue_ptr, num_samples_in_queue):

                samples_queue, queue_ptr, num_samples_in_queue = (
                    tf.cond(pred=tf.equal(num_samples_in_queue, 0),
                            true_fn=lambda: (tf.random.shuffle(samples_indcs),
                                             tf.constant(0),
                                             queue_capacity),
                            false_fn=lambda: (samples_queue, queue_ptr, num_samples_in_queue)))

                '''if num_samples_in_queue == tf.constant(0):
                    # shuffle and refill the queue
                    samples_queue = tf.random.shuffle(class_samples_indcs[k])
                    queue_ptr = tf.constant(0)
                    num_samples_in_queue = queue_capacity
                else:
                    samples_queue = samples_queue
                    queue_ptr = queue_ptr
                    num_samples_in_queue = num_samples_in_queue'''

                # pick next sample in the queue
                picked_indcs = picked_indcs.write(p, samples_queue[queue_ptr])
                #self.picked_indcs_per_class.scatter_nd_update(indices=[[k, p]], updates=[samples_queue[que_ptr]])
                queue_ptr += 1
                num_samples_in_queue -= 1
                p += 1

                return p, picked_indcs, samples_queue, queue_ptr, num_samples_in_queue

            def p_cond(p, picked_indcs, samples_queue, queue_ptr, num_samples_in_queue):
                return tf.less(p, total_picks_per_class)

            p, picked_indcs, samples_queue, queue_ptr, num_samples_in_queue = (
                tf.while_loop(cond=p_cond, body=pick_from_shuffle_queue,
                              loop_vars=(p_0, picked_indcs_0, samples_queue_0, queue_ptr_0, num_samples_in_queue_0),
                              parallel_iterations=1)) # to control randomness

            picked_indcs_per_class = picked_indcs_per_class.write(k, picked_indcs.stack())

            k += 1

            return k, picked_indcs_per_class

        def k_cond(k, picked_indcs_per_class):
            return tf.less(k, self.num_classes)

        # realization of the for loop
        k_0 = tf.constant(0)
        picked_indcs_per_class_0 = (
            tf.TensorArray(dtype=tf.int32, size=self.num_classes, dynamic_size=False,
                           clear_after_read=False,
                           element_shape=tf.TensorShape([total_picks_per_class])))

        k, picked_indcs_per_class = (
            tf.while_loop(cond=k_cond, body=shuffle_and_pick_m,
                          loop_vars=(k_0, picked_indcs_per_class_0)))

        picked_indcs_per_class = picked_indcs_per_class.stack()

        # each row corresponds to uniform samples of a particular class
        picked_samples = tf.gather(class_specific_datasets_as_tensor, picked_indcs_per_class, batch_dims=1)
        m_samples_per_class_list = tf.split(picked_samples, self.pick_repeats, axis=1)
        # make sure that each class is visited once prior to any class is sampled again
        if self.shuffle_classes == True:
            samples_for_epoch = tf.concat([tf.random.shuffle(t) for t in m_samples_per_class_list], axis=0)
        else:
            #tf.print('\nproxy sampler with repeats: {}'.format(self.pick_repeats))
            samples_for_epoch = tf.concat(m_samples_per_class_list, axis=0)

        return tf.data.Dataset.from_tensor_slices(tf.squeeze(samples_for_epoch))

# base class for samplers
class Sampler(object):
    '''def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            '__init__() is to be implemented in Sampler sub classes')'''

    def __call__(self, dataset):
        raise NotImplementedError(
            '__call__() is to be implemented in Sampler sub classes')

# random samping
class Random(Sampler):
    def __init__(self, batch_size=32, shuffling_buffer_size=10000, random_seed=None, **kwargs):
        super(Random, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.shuffling_buffer_size = shuffling_buffer_size
        self.random_seed = random_seed

    def __call__(self, dataset):

        dataset = tf.data.Dataset.from_tensor_slices([ds.strip() for ds in dataset])

        dataset = dataset.shuffle(buffer_size=self.shuffling_buffer_size, seed=self.random_seed).repeat()

        dataset = dataset.batch(self.batch_size)

        return dataset

# random sampling with classes are to follow the same prior probability
class RandomUniformApriori(Sampler):
    def __init__(self, batch_size=32, random_seed=102, **kwargs):
        super(RandomUniformApriori, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.random_seed = random_seed

    def __call__(self, dataset):

        dataset_size = len(dataset)

        # determine number of classes
        max_id = -1
        for d in dataset:
            l = int(d.split()[0])
            if l > max_id:
                max_id = l
        num_classes = max_id + 1  # since class-k has label k-1

        # divide datasets into classes
        # datasets is of the form <label> <img_path> <height> <width> <bbox:ymin> <bbox:xmin> <bbox:ymax> <bbox:xmax>
        class_specific_datasets = [[] for _ in range(num_classes)]
        [class_specific_datasets[int(d.split()[0])].append(d.strip()) for d in dataset]

        '''# divide datasets into classes
        # datasets is of the form (data, label)
        class_specific_datasets = [[] for _ in range(self.num_classes)]
        [class_specific_datasets[int(d[1])].append(d[0]) for d in dataset]'''

        class_specific_datasets = tf.ragged.constant(class_specific_datasets)

        all_in_one = tf.data.Dataset.from_tensors(class_specific_datasets)

        # each item of shape=(1, num_sample_per_class) and each item is of different class
        uniform_samples = all_in_one.flat_map(DatasetToRandomSamplesPerClass(num_classes=num_classes,
                                                                             sample_per_class=1,
                                                                             dataset_size=dataset_size,
                                                                             shuffle_classes=True)).repeat()


        return uniform_samples.batch(self.batch_size).map(lambda x:tf.reshape(x, shape=(-1,)))

# sampling M samples per class (sampling is without replacement)
class MPerClass(Sampler):
    def __init__(self, classes_per_batch, sample_per_class,
                 representative_lifetime=0,
                 preprocess_representative=False,
                 shuffle_classes=True,
                 **kwargs):
        '''if representaive_lifetime > 0 this method samples representatives for each class and repeats those and
        places an auxillary label to indicate whether the sample is representative (0) or data (1)
        and this can be used in preprocessing'''

        super(MPerClass, self).__init__(**kwargs)


        tf.assert_greater(sample_per_class, 0,
                          message="number of samples per class should be positive")

        batch_size = classes_per_batch * sample_per_class

        if representative_lifetime > 0:
            tf.assert_greater(sample_per_class, 1,
                              message="sample per class should be at least two for representative sampling.")
            sample_per_class -= 1

        self.classes_per_batch = classes_per_batch
        self.sample_per_class = sample_per_class
        self.representative_lifetime = representative_lifetime
        self.preprocess_representative = preprocess_representative
        self.batch_size = batch_size

    #@tf.function
    def __call__(self, dataset):

        dataset_size = len(dataset)

        # determine number of classes
        max_id = -1
        for d in dataset:
            l = int(d.split()[0])
            if l > max_id:
                max_id = l
        num_classes = max_id + 1 # since class-k has label k-1

        tf.assert_greater(num_classes + 1, self.classes_per_batch,
                          message="number of classes to be picked cannot be greater than the total number of classes.")

             # divide datasets into classes
        # datasets is of the form <label> <img_path> <height> <width> <bbox:ymin> <bbox:xmin> <bbox:ymax> <bbox:xmax>
        class_specific_datasets = [[] for _ in range(num_classes)]
        [class_specific_datasets[int(d.split()[0])].append(d.strip()) for d in dataset]


        '''# datasets is of the form (data, label)
        class_specific_datasets = [[] for _ in range(self.num_classes)]
        [class_specific_datasets[int(d[1])].append(d[0]) for d in dataset]'''

        class_specific_datasets = tf.ragged.constant(class_specific_datasets)

        all_in_one = tf.data.Dataset.from_tensors(class_specific_datasets)

        # each item of shape=(1, num_sample_per_class) and each item is of different class
        m_samples_per_class = all_in_one.flat_map(DatasetToRandomSamplesPerClass(num_classes=num_classes,
                                                                                 sample_per_class=self.sample_per_class,
                                                                                 dataset_size=dataset_size,
                                                                                 shuffle_classes=True)).repeat()

        # batch has <num_classes_to_pick> classes with <num_sample_per_class> samples each
        m_samples_per_class = (
            m_samples_per_class.batch(self.classes_per_batch).map(lambda x:
                                                                    tf.reshape(tf.transpose(x), shape=(-1, ))))
        # batch structure is: s_i1, s_j1, s_k1, ..., s_i2, s_j2, s_k2, ... where s_in is the nth sample of class i

        if self.representative_lifetime > 0:
            a_sample_per_class_in_sort = (
                all_in_one.flat_map(DatasetToRandomSamplesPerClass(num_classes=num_classes,
                                                                   sample_per_class=1,
                                                                   dataset_size=dataset_size,
                                                                   shuffle_classes=False)).repeat())

            class_representatives = a_sample_per_class_in_sort.batch(num_classes)

            class_representatives = (
                class_representatives.flat_map(lambda x:
                                               tf.data.Dataset.from_tensors(x).repeat(self.representative_lifetime)))

            def append_class_representatives(representatives, m_per_class_batch):
                indcs = tf.stack(
                    [tf.strings.to_number(tf.strings.split(m_per_class_batch[k], ' ')[0], out_type=tf.int32)
                     for k in range(self.classes_per_batch)])

                reps = tf.gather(representatives, indcs)

                preproc = int(not self.preprocess_representative)
                aux_labels = [preproc] * self.classes_per_batch + [0] * self.classes_per_batch * self.sample_per_class

                return (tf.concat([reps, m_per_class_batch], axis=0), tf.constant(aux_labels, dtype=tf.int32))


            m_samples_per_class = (
                tf.data.Dataset.zip((class_representatives, m_samples_per_class)).map(append_class_representatives))


        return m_samples_per_class

# sampling 1 sample per class (sampling is without replacement)
class OnePerClass(Sampler):
    def __init__(self, batch_size=None, **kwargs):
        '''if representaive_lifetime > 0 this method samples representatives for each class and repeats those and
        places an auxillary label to indicate whether the sample is representative (0) or data (1)
        and this can be used in preprocessing'''

        super(OnePerClass, self).__init__(**kwargs)



        self.batch_size = batch_size


    #@tf.function
    def __call__(self, dataset):

        dataset_size = len(dataset)

        # determine number of classes (assuming labels are consecutive starting at 0)
        max_id = -1
        for d in dataset:
            l = int(d.split()[0])
            if l > max_id:
                max_id = l
        num_classes = max_id + 1 # since class-k has label k-1


             # divide datasets into classes
        # datasets is of the form <label> <img_path> <height> <width> <bbox:ymin> <bbox:xmin> <bbox:ymax> <bbox:xmax>
        class_specific_datasets = [[] for _ in range(num_classes)]
        [class_specific_datasets[int(d.split()[0])].append(d.strip()) for d in dataset]


        '''# datasets is of the form (data, label)
        class_specific_datasets = [[] for _ in range(self.num_classes)]
        [class_specific_datasets[int(d[1])].append(d[0]) for d in dataset]'''

        class_specific_datasets = tf.ragged.constant(class_specific_datasets)

        all_in_one = tf.data.Dataset.from_tensors(class_specific_datasets)

        # each item of shape=(1, 1) and each item is of different class
        one_sample_per_class = all_in_one.flat_map(DatasetToRandomSamplesPerClass(num_classes=num_classes,
                                                                                 sample_per_class=1,
                                                                                 dataset_size=dataset_size,
                                                                                 shuffle_classes=False)).repeat()

        # each batch has num_classes elements (i.e. a sample from each distict class)
        sample_from_each = one_sample_per_class.batch(num_classes).map(lambda x: tf.reshape(x, shape=(-1,)))

        # a sample from each class will be drawn in order to small batches. last batch can be smaller
        # eg. (10 classes and batch_size 4) [0,1,2,3], [4,5,6,7], [8,9], [0,1,2,3], ...
        sample_from_each_in_batches = (
            sample_from_each.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x).batch(self.batch_size)))


        return sample_from_each_in_batches



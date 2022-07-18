import tensorflow as tf

import numpy as np

import os


import cv2


#from .dataset_base import Dataset

#from matplotlib import pyplot as plt

sub_classes = [
    ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    ['bottle', 'bowl', 'can', 'cup', 'plate'],
    ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    ['crab', 'lobster', 'snail', 'spider', 'worm'],
    ['baby', 'boy', 'girl', 'man', 'woman'],
    ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

super_classes = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers',
    'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture',
    'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores', 'medium-sized_mammals', 'non-insect_invertebrates', 'people',
    'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']

class_dict = dict()
for s, c in zip(super_classes, sub_classes):
    class_dict[s] = c


def create_collage_sources(image_path, save_dir):

    target_classes = [
        'aquatic_mammals', 'fish', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_omnivores_and_herbivores', 'medium-sized_mammals', 'non-insect_invertebrates', 'people',
        'reptiles', 'small_mammals', 'vehicles_1', 'vehicles_2']

    background_classes = ['large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'trees', 'flowers']

    train_img_list = []
    test_img_list = []
    for super_class in background_classes:

        for category in class_dict[super_class][:3]:
            category_folder = os.path.join(image_path, category)
            train_img_list += [
                cv2.imread(os.path.join(category_folder, fname), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
                for fname in
                [x for x in os.listdir(category_folder) if x.endswith('.png')]]


        for category in class_dict[super_class][3:]:
            category_folder = os.path.join(image_path, category)
            test_img_list += [
                cv2.imread(os.path.join(category_folder, fname), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
                for fname in
                [x for x in os.listdir(category_folder) if x.endswith('.png')]]



    bg_train_imgs = np.stack(train_img_list, axis=0)
    np.random.shuffle(bg_train_imgs)
    train_bg_size = bg_train_imgs.shape[0]

    bg_test_imgs = np.stack(test_img_list, axis=0)
    np.random.shuffle(bg_test_imgs)
    test_bg_size = bg_test_imgs.shape[0]

    class_collage_train_list = []
    class_collage_test_list = []
    train_bg_pointer = 0
    test_bg_pointer = 0
    placement_map = {0: (0, 0), 1: (0, 32), 2: (32, 0), 3: (32, 32)}
    for label, super_class in enumerate(target_classes):

        img_list = []
        for category in class_dict[super_class][:3]:
            category_folder = os.path.join(image_path, category)

            for fname in [x for x in os.listdir(category_folder) if x.endswith('.png')]:
                placements = np.random.permutation(np.arange(4))
                cate_img = cv2.imread(os.path.join(category_folder, fname), cv2.IMREAD_UNCHANGED)[:, :, ::-1]

                if (train_bg_pointer + 3) > train_bg_size:
                    np.random.shuffle(bg_train_imgs)
                    train_bg_pointer = 0

                collage_img = np.zeros(shape=(64, 64, 3), dtype=np.uint8)

                r_beg = placement_map[placements[0]][0]
                c_beg = placement_map[placements[0]][1]
                collage_img[r_beg:r_beg+32, c_beg:c_beg+32, :] = cate_img
                for p in placements[1:]:
                    r_beg = placement_map[p][0]
                    c_beg = placement_map[p][1]
                    collage_img[r_beg:r_beg + 32, c_beg:c_beg + 32, :] = bg_train_imgs[train_bg_pointer]
                    train_bg_pointer += 1

                img_list.append(collage_img)

        class_collage_train_list.append(np.stack(img_list, axis=0))

        img_list = []
        for category in class_dict[super_class][3:]:
            category_folder = os.path.join(image_path, category)

            for fname in [x for x in os.listdir(category_folder) if x.endswith('.png')]:
                placements = np.random.permutation(np.arange(4))
                cate_img = cv2.imread(os.path.join(category_folder, fname), cv2.IMREAD_UNCHANGED)[:, :, ::-1]

                if (test_bg_pointer + 3) > test_bg_size:
                    np.random.shuffle(bg_test_imgs)
                    test_bg_pointer = 0

                collage_img = np.zeros(shape=(64, 64, 3), dtype=np.uint8)

                r_beg = placement_map[placements[0]][0]
                c_beg = placement_map[placements[0]][1]
                collage_img[r_beg:r_beg + 32, c_beg:c_beg + 32, :] = cate_img

                for p in placements[1:]:
                    r_beg = placement_map[p][0]
                    c_beg = placement_map[p][1]
                    collage_img[r_beg:r_beg + 32, c_beg:c_beg + 32, :] = bg_test_imgs[test_bg_pointer]
                    test_bg_pointer += 1

                img_list.append(collage_img)

        class_collage_test_list.append(np.stack(img_list, axis=0))

    train_data = np.stack(class_collage_train_list, axis=0)
    np.save(os.path.join(save_dir, 'train_data.npy'), train_data)

    test_data = np.stack(class_collage_test_list, axis=0)
    np.save(os.path.join(save_dir, 'test_data.npy'), test_data)

    return train_data, test_data

def create_distilled_sources(image_path, save_dir):

    target_classes = [
        'aquatic_mammals', 'fish', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_omnivores_and_herbivores', 'medium-sized_mammals', 'non-insect_invertebrates', 'people',
        'reptiles', 'small_mammals', 'vehicles_1', 'vehicles_2']


    class_train_list = []
    class_test_list = []

    for label, super_class in enumerate(target_classes):

        img_list = []
        for category in class_dict[super_class][:3]:
            category_folder = os.path.join(image_path, category)

            img_list += [cv2.imread(os.path.join(category_folder, fname), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
                         for fname in [x for x in os.listdir(category_folder) if x.endswith('.png')]]

        class_train_list.append(np.stack(img_list, axis=0))

        img_list = []
        for category in class_dict[super_class][3:]:
            category_folder = os.path.join(image_path, category)

            img_list += [cv2.imread(os.path.join(category_folder, fname), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
                         for fname in [x for x in os.listdir(category_folder) if x.endswith('.png')]]

        class_test_list.append(np.stack(img_list, axis=0))

    train_data = np.stack(class_train_list, axis=0)
    np.save(os.path.join(save_dir, 'train_data.npy'), train_data)

    test_data = np.stack(class_test_list, axis=0)
    np.save(os.path.join(save_dir, 'test_data.npy'), test_data)

    return train_data, test_data


image_path = '/home/ogam3080/Desktop/yeti/metric_learning/datasets/Cifar100ML/cifar100/data'
save_dir = '/home/ogam3080/Desktop/yeti/metric_learning/datasets/CifarDistilled'

#train_data, test_data = create_collage_sources(image_path, save_dir)
#train_data, test_data = create_distilled_sources(image_path, save_dir)

class CifarCollage(object):

    def __init__(self, dataset_dir):


        self.train_data = tf.constant(np.load(os.path.join(dataset_dir, 'train_data.npy')))
        self.test_data = tf.constant(np.load(os.path.join(dataset_dir, 'test_data.npy')))


    def createDataset(self, samples_per_class=4, test_batch_size=192):

        @tf.function
        def m_per_class(all_in_one_dataset):
            ''' the input is of shape=(num_classes, samples_per_class, width, height, channels) '''

            num_classes = 16
            samples_for_epoch = tf.concat([tf.random.shuffle(t) for t in
                                           tf.split(
                                               tf.transpose(all_in_one_dataset, perm=(1, 0, 2, 3, 4)),
                                               num_classes, axis=1)], axis=1)

            return tf.data.Dataset.from_tensor_slices(samples_for_epoch).batch(samples_per_class)



        @tf.function
        def to_batch(x):
            ''' the input is of shape=(m, num_classes, width, height, channels) '''
            num_classes = 16
            _, _, w, h, c = x.shape.as_list()
            imgs = tf.reshape(x, shape=(samples_per_class * num_classes, w, h, c))
            imgs = tf.cast(imgs, dtype=tf.float32) / 255. - 0.5

            labels = tf.tile(tf.range(num_classes), [samples_per_class])

            return imgs, labels


        train_data_epoch = tf.data.Dataset.from_tensors(self.train_data).flat_map(m_per_class).repeat()
        train_batches = train_data_epoch.map(to_batch)

        num_classes, num_samples, w, h, c = self.test_data.shape.as_list()
        test_data = tf.reshape(tf.transpose(self.test_data, (1, 0, 2, 3, 4)),
                               shape=(num_classes * num_samples, w, h, c))
        test_labels = tf.tile(tf.range(num_classes), [num_samples])

        test_batches = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(
            test_batch_size) .map(lambda *x: (tf.cast(x[0], dtype=tf.float32) / 255. - 0.5, x[1]))


        return train_batches, test_batches
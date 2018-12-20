"Author:MingBo Hong"
import shutil
import tensorflow as tf
from scipy import misc
import os
import numpy as np
import math
import  train

C_size = 32
dir = ['Biometrika', 'CrossMatch', 'Identix']
classes = ['Alive', 'Silicone', 'Gelatin', 'PlayDoh'] #     """多种类别，但是我们默认Alive:True，other：False"""
Binary = ['img_fake','img_true']
Binary_dir = ['Fake','True']

def shuffle(image_path,record_name):
    data = []
    writer = tf.python_io.TFRecordWriter(image_path + "/" + str(record_name))
    for index, name in enumerate(Binary_dir):
        try:
            sample_path = os.path.join(image_path, name)
            for lists in os.listdir(sample_path):
                sub_path = os.path.join(sample_path, lists)
                pic = misc.imread(sub_path)
                data.append([pic,index])
        except FileNotFoundError:
            print("File is not found.")
    data = np.array(data)
    np.random.shuffle(data)
    train_data = np.array([x[0] for x in data])
    train_label = np.array([x[1] for x in data])
    for i in range(data.shape[0]):
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[train_label[i]])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_data[i].tobytes()]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


#补足图片大小，使得按照 C_size*C_size 大小去分割，并且没有重叠部分
def pad_image(ori_image):

    if math.ceil(ori_image.shape[0] / C_size) > int(ori_image.shape[0] / C_size):
        pad_height = math.ceil(ori_image.shape[0] / C_size)*32 - ori_image.shape[0]
    else:
        pad_height = 0
    if math.ceil(ori_image.shape[1]/C_size) > int(ori_image.shape[1]/C_size):
        pad_width = math.ceil(ori_image.shape[1] / C_size)*32 - ori_image.shape[1]
    else:
        pad_width = 0
    return np.pad(ori_image, ((math.ceil(pad_height/2), int(pad_height/2)), (math.ceil(pad_width/2), int(pad_width/2))), 'constant', constant_values=255)



def transform_data(image_path,record_name):
    writer = tf.python_io.TFRecordWriter(image_path+"/"+str(record_name))
    for index, name in enumerate(Binary_dir):
        try:
            sample_path = os.path.join(image_path, name)
            for lists in os.listdir(sample_path):
                sub_path = os.path.join(sample_path, lists)
                pic = misc.imread(sub_path)
                k = k+1
                img_raw = pic.tobytes()  # 将图片转化为原生byte
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
        except FileNotFoundError:
            print("File is not found.")

    writer.close()




def cut_image(image,image_dir,count):
    k = 0
    #"分割图片，并且统计分割后图片的均值方差，以及设置相应的阈值，进行分割保存"
    var = 0
    number_slice = (image.shape[0]* image.shape[1])/(C_size*C_size)
    mean  = np.mean(image)
    for row in range(int(image.shape[0] / C_size)):
        for col in range(int((image.shape[1]) / C_size)):
            row_start = row * C_size
            row_end = (row+1) * C_size
            col_start = col * C_size
            col_end = (col+1) * C_size
            var = var + np.var(image[ row_start:row_end, col_start:col_end])
    for row in range(int(image.shape[0] /C_size)):
        for col in range(int((image.shape[1]) /C_size)):
            row_start = row * C_size
            row_end = (row + 1) * C_size
            col_start = col * C_size
            col_end = (col + 1) * C_size

            if  np.var(image[row_start:row_end, col_start:col_end]) > (var / number_slice) * 0.5 \
                    and np.mean(image[row_start:row_end, col_start:col_end])< mean*0.9:
                misc.imsave(image_dir + "/"+str(count)+"_{}.jpg".format(k), image[row_start:row_end, col_start:col_end])
                k+=1

def load_data(path):
    for index, name in enumerate(Binary):
            count = 0
            sample_path = os.path.join(path, name)
            if name == 'img_fake':
                dir = os.path.join(path, 'Fake')
            else:
                dir = os.path.join(path, 'True')
            if not os.path.exists(dir):
                os.makedirs(dir)
            try:
                for lists in os.listdir(sample_path):
                    sub_path = os.path.join(sample_path, lists)
                    image = misc.imread(sub_path)
                    if len(image.shape) == 3:
                        image = rgb2gray(image)
                        image = np.squeeze(image)
                    pic = pad_image(image)
                    try:
                        cut_image(pic, dir, count)
                        count +=1
                    except:
                        print("Error：process picture：",lists)
            except FileNotFoundError:
                print("File is not found.")

def copy_data(path,copy_dir):
    Alive = 0
    Fake = 0
    if not os.path.exists(os.path.join(copy_dir, "img_True")):
        os.makedirs(os.path.join(copy_dir, "img_true"))
    if not os.path.exists(os.path.join(copy_dir, "img_Fake")):
        os.makedirs(os.path.join(copy_dir, "img_Fake"))

    for _, dir_name in enumerate(dir):
        if dir_name == 'Biometrika':
            for index, name in enumerate(classes):
                img_path = os.path.join(path, dir_name)
                sample_path = os.path.join(img_path, name)
                try:
                    for lists in os.listdir(sample_path):
                            sub_path = os.path.join(sample_path, lists)
                            if name =="Alive":
                                Alive = Alive + 1
                                shutil.copy(sub_path, os.path.join(copy_dir, "img_True")+"\{}.jpg".format(Alive))
                            else:
                                Fake = Fake + 1
                                shutil.copy(sub_path, os.path.join(copy_dir, "img_Fake")+"\{}.jpg".format(Fake))

                except FileNotFoundError:
                    print("File is not found.")
        else:
            for index, name in enumerate(classes):

                img_path = os.path.join(path, dir_name)
                sample_path = os.path.join(img_path, name)
                try:
                    for lists in os.listdir(sample_path):
                        for pic in os.listdir(os.path.join(sample_path, lists)):
                            sub_path = os.path.join(os.path.join(sample_path, lists), pic)
                            if name == "Alive":
                                Alive = Alive + 1
                                shutil.copy(sub_path, os.path.join(copy_dir, "img_True") + "\{}.jpg".format(Alive))
                            else:
                                Fake = Fake + 1
                                shutil.copy(sub_path, os.path.join(copy_dir, "img_Fake") + "\{}.jpg".format(Fake))
                except FileNotFoundError:
                    print("File is not found.")





#训练过程中，从队列中获得一个随机打乱或不打乱的batch_size 样本
def get_batch_samples(img_obj, batch_size, shuffle_flag):

    if shuffle_flag == False:
        image_batch, label_batch = tf.train.batch(tensors=img_obj,
                                                  batch_size=batch_size,
                                                  num_threads=1,
                                                  capacity=512)
    else:
        image_batch, label_batch = tf.train.shuffle_batch(tensors=img_obj,
                                                          batch_size=batch_size,
                                                          num_threads=1,
                                                          min_after_dequeue=128,
                                                          capacity=512)

    tf.summary.image("images", image_batch)
    return image_batch, tf.reshape(label_batch, shape=[batch_size])






def read_and_decode(filename,batch_size,flag):
    #根据文件名生成一个队列
    if flag:
        filename_queue = tf.train.string_input_producer([filename],num_epochs=train.epochs)
    else:
        filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [C_size, C_size, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return get_batch_samples([img, label], batch_size, shuffle_flag=flag)

def process(training_img_path,imagedir,name):
    #copy_data(path=training_img_path,copy_dir=imagedir)  #将解压后的数据全部拷贝到copy_dir目录下，并且自动分为二类，自动归档
    load_data(path=imagedir)  # 加载Binary，并且自动分割保存
    shuffle(imagedir,name)

if __name__ == '__main__':
    # #===================trian======================================
    #imagedir = r"F:\KDR\BRL\project\Train_image"      #分割后图片的路径
    #training_img_path = r"G:\BRL\pF:\KDR\BRL\project\LivDet2009\Training"  #训练集图片路径
    # #=======================================================#
    # ===================test======================================
    imagedir = r"F:\KDR\BRL\project\Test_image"  # 分割后图片的路径
    training_img_path = r"F:\KDR\BRL\project\LivDet2009\Testing"  # 训练集图片路径
    process(training_img_path,imagedir,'Test.tfrecords')





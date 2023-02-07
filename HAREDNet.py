import os
import glob 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
import gc 
import random
import pathlib
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, LeakyReLU
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
from keras import regularizers
import keras.backend as kb
import shutil

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 5.0) # set default size of plots


keras = tf.keras
layers = tf.keras.layers

rgb_dir = '/kaggle/input/hmdb51/HMDB51/'
rgb_root = pathlib.Path(rgb_dir)

def load_file(path):
    train = np.load(path)
    train = train.tolist()
    random.shuffle(train)
    train_ps=[]
    l1=[]
    l2=[]
    l3=[]
    for p in train:
        rt = pathlib.Path(p)
        ps = sorted(list(rt.glob('*.jpg')))
        num = len(ps)
        step = num//3
        for i in range(5):
            num1 = random.randint(0,step-1)
            num2 = random.randint(step,step*2-1)
            num3 = random.randint(step*2,num-1)
            l1.append(str(ps[num1]))
            l2.append(str(ps[num2]))
            l3.append(str(ps[num3]))
    l = list((l1,l2,l3))
    return l

train_ps = load_file('/kaggle/input/hmdblist/hmdb_train.npy')
test_ps = load_file('/kaggle/input/hmdblist/hmdb_test.npy')
train_count = len(train_ps[0])
test_count = len(test_ps[0])
image_count = train_count + test_count
print(image_count)

label_names = sorted(item.name for item in rgb_root.glob('*/') if item.is_dir())
label_names

label_to_index = dict((name, index) for index,name in enumerate(label_names))
label_to_index

train_labels = [label_to_index[pathlib.Path(path).parent.parent.name] for path in train_ps[0]]
test_labels = [label_to_index[pathlib.Path(path).parent.parent.name] for path in test_ps[0]]

train_label_onehot = tf.keras.utils.to_categorical(train_labels)
test_label_onehot = tf.keras.utils.to_categorical(test_labels)

def load_preprosess_image(path):
    a = [256,224,192,168]
    n1 = random.randint(0,3)
    n2 = random.randint(0,3)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.resize(image, [256, 340])
    image = tf.image.random_contrast(image, 0.6, 1)
    image = tf.image.random_crop(image, [a[n1], a[n2], 3])
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image/255
    return image
def load_preprosess_image1(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image,[224,224])
    image = tf.cast(image, tf.float32)
    image = image/255
    return image

AUTOTUNE = tf.data.experimental.AUTOTUNE
def path2image(paths,fn):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    image_ds = path_ds.map(fn)
    return image_ds
def image_label(paths,lot,fn):
    image_ds1 = path2image(paths[0],fn)
    image_ds2 = path2image(paths[1],fn)
    image_ds3 = path2image(paths[2],fn)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(lot, tf.int64))
    image_ds = tf.data.Dataset.zip((image_ds1, image_ds2,image_ds3))
    image_label_ds = tf.data.Dataset.zip((image_ds,label_ds))
    return image_label_ds

def load_image_from_dir(img_path):
    file_list = glob.glob(img_path+'/*.png')
    file_list.sort()
    img_list = np.empty((len(file_list), target_height, target_width, 1))
    for i, fig in enumerate(file_list):
        img = image.load_img(fig, color_mode='grayscale', target_size=(target_height, target_width))
        img_array = image.img_to_array(img).astype('float32')
        img_array = img_array / 255.0
        img_list[i] = img_array
    
    return img_list

def train_test_split(data,random_seed=55,split=0.75):
    set_rdm = np.random.RandomState(seed=random_seed)
    dsize = len(data)
    ind = set_rdm.choice(dsize,dsize,replace=False)
    train_ind = ind[:int(0.75*dsize)]
    val_ind = ind[int(0.75*dsize):]
    return data[train_ind],data[val_ind]

def augment_pipeline(pipeline, images, seed=5):
    ia.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = np.array(step.augment_images(images))
        processed_images = np.append(processed_images, temp, axis=0)
    return(processed_images)

train_data = image_label(train_ps,train_label_onehot,load_preprosess_image)
test_data = image_label(test_ps,test_label_onehot,load_preprosess_image1)

BATCH_SIZE = 256

full_train = train_data.repeat(-1).shuffle(BATCH_SIZE*2+50).batch(BATCH_SIZE)
full_target = test_data.repeat(-1).batch(BATCH_SIZE)

rotate90 = iaa.Rot90(1) # rotate image 90 degrees
rotate180 = iaa.Rot90(2) # rotate image 180 degrees
rotate270 = iaa.Rot90(3) # rotate image 270 degrees
random_rotate = iaa.Rot90((1,3)) # randomly rotate image from 90,180,270 degrees
perc_transform = iaa.PerspectiveTransform(scale=(0.02, 0.1)) # Skews and transform images without black bg
rotate10 = iaa.Affine(rotate=(10)) # rotate image 10 degrees
rotate10r = iaa.Affine(rotate=(-10)) # rotate image 30 degrees in reverse
crop = iaa.Crop(px=(5, 32)) # Crop between 5 to 32 pixels
hflip = iaa.Fliplr(1) # horizontal flips for 100% of images
vflip = iaa.Flipud(1) # vertical flips for 100% of images
gblur = iaa.GaussianBlur(sigma=(1, 1.5)) # gaussian blur images with a sigma of 1.0 to 1.5
motionblur = iaa.MotionBlur(8) # motion blur images with a kernel size 8

seq_rp = iaa.Sequential([
    iaa.Rot90((1,3)), # randomly rotate image from 90,180,270 degrees
    iaa.PerspectiveTransform(scale=(0.02, 0.1)) # Skews and transform images without black bg
])

seq_cfg = iaa.Sequential([
    iaa.Crop(px=(5, 32)), # crop images from each side by 5 to 32px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 1.5)) # blur images with a sigma of 0 to 1.5
])

seq_fm = iaa.Sequential([
    iaa.Flipud(1), # vertical flips all the images
    iaa.MotionBlur(k=6) # motion blur images with a kernel size 6
])

pipeline = []
pipeline.append(rotate90)
pipeline.append(rotate180)
pipeline.append(rotate270)
# pipeline.append(random_rotate)
pipeline.append(perc_transform)
# pipeline.append(rotate10)
# pipeline.append(rotate10r)
pipeline.append(crop)
pipeline.append(hflip)
pipeline.append(vflip)
# pipeline.append(gblur)
# pipeline.append(motionblur)
pipeline.append(seq_rp)
pipeline.append(seq_cfg)
pipeline.append(seq_fm)

%%time
processed_train = augment_pipeline(pipeline, full_train.reshape(-1,target_height,target_width))
processed_target = augment_pipeline(pipeline, full_target.reshape(-1,target_height,target_width))

processed_train = processed_train.reshape(-1,target_height,target_width,1)
processed_target = processed_target.reshape(-1,target_height,target_width,1)

processed_train.shape

train, val = train_test_split(processed_train, random_seed=9, split=0.8)
target_train, target_val = train_test_split(processed_target, random_seed=9, split=0.8)

# train, val = train_test_split(full_train, random_seed=9, split=0.8)
# target_train, target_val = train_test_split(full_target, random_seed=9, split=0.8)

# time
# pre_train, pre_val = train_test_split(full_train, random_seed=9, split=0.7)
# pre_target_train, pre_target_val = train_test_split(full_target, random_seed=9, split=0.7)

# print(pre_train.shape,pre_val.shape)

# train = augment_pipeline(pipeline, pre_train.reshape(-1,target_height,target_width), seed=10)
# target_train = augment_pipeline(pipeline, pre_target_train.reshape(-1,target_height,target_width), seed=10)

# train = train.reshape(-1,target_height,target_width,1)
# target_train = target_train.reshape(-1,target_height,target_width,1)

# val_pipeline = pipeline + [seq_fm]

# val = augment_pipeline(val_pipeline, pre_val.reshape(-1,target_height,target_width))
# target_val = augment_pipeline(val_pipeline, pre_target_val.reshape(-1,target_height,target_width))

# val = val.reshape(-1,target_height,target_width,1)
# target_val = target_val.reshape(-1,target_height,target_width,1)

# print("Shape of Train set:",train.shape)
# print("Shape of Validation set:",val.shape)

# optimizer = Adam(lr=1e-2, decay=1e-7)
# optimizer = Adam(lr=0.001)
# l2 = 0.01

# ### Multi Layer auto encoder
# input_layer = Input(shape=train[0].shape)

# # encoder
# e = Conv2D(96, (3, 3), activation='relu', padding='same')(input_layer)
# e = Conv2D(96, (3, 3), activation='relu', padding='same')(e)
# e = Conv2D(64, (3, 3), activation='relu', padding='same')(e)
# # e = MaxPooling2D((2, 2), padding='same')(e)

# # decoder
# d = Conv2D(64, (3, 3), activation='relu', padding='same')(e)
# d = Conv2D(96, (3, 3), activation='relu', padding='same')(d)
# d = Conv2D(96, (3, 3), activation='relu', padding='same')(d)
# # d = UpSampling2D((2, 2))(d)
# output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)


# ### Simple Auto encoder

# input_layer = Input(shape=train[0].shape)

# # encoder
# e1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
# e2 = Conv2D(32, (2, 2), activation='relu', padding='same')(e1)
# eb = MaxPooling2D((2, 2), padding='same')(e2)

# # decoder
# d1 = Conv2D(32, (2, 2), activation='relu', padding='same')(eb)
# d2 = Conv2D(64, (3, 3), activation='relu', padding='same')(d1)
# db = UpSampling2D((2, 2))(d2)
# output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(db)

# ### Multi layer auto encoder with regularization
# input_layer = Input(shape=train[0].shape)

# # encoder
# e = Conv2D(128, (3, 3), activation='relu', activity_regularizer=regularizers.l2(l2), padding='same')(input_layer)
# e = Conv2D(128, (3, 3), activation='relu', activity_regularizer=regularizers.l2(l2), padding='same')(e)
# e = Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l2(l2), padding='same')(e)
# e = MaxPooling2D((2, 2), padding='same')(e)

# # decoder
# d = Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l2(l2), padding='same')(e)
# d = UpSampling2D((2, 2))(d)
# d = Conv2D(128, (3, 3), activation='relu', activity_regularizer=regularizers.l2(l2), padding='same')(d)
# d = Conv2D(128, (3, 3), activation='relu', activity_regularizer=regularizers.l2(l2), padding='same')(d)
# output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

### Multi layer auto encoder with LeakyRelu and Normalization
input_layer = Input(shape=(None,None,1))

# encoder
e = Conv2D(32, (3, 3), padding='same')(input_layer)
e = LeakyReLU(alpha=0.3)(e)
e = BatchNormalization()(e)
e = Conv2D(64, (3, 3), padding='same')(e)
e = LeakyReLU(alpha=0.3)(e)
e = BatchNormalization()(e)
e = Conv2D(64, (3, 3), padding='same')(e)
e = LeakyReLU(alpha=0.3)(e)
e = MaxPooling2D((2, 2), padding='same')(e)

# decoder
d = Conv2D(64, (3, 3), padding='same')(e)
d = LeakyReLU(alpha=0.3)(d)
d = BatchNormalization()(d)

d = Conv2D(64, (3, 3), padding='same')(d)
d = LeakyReLU(alpha=0.3)(d)
# e = BatchNormalization()(e)
d = UpSampling2D((2, 2))(d)
d = Conv2D(32, (3, 3), padding='same')(d)
d = LeakyReLU(alpha=0.2)(d)
# d = Conv2D(128, (3, 3), padding='same')(d)
output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

# optimizer = Adam(lr=1e-4, decay=7e-6)
optimizer = Adam(lr=9e-4, decay=1e-5)
AEmodel = Model(input_layer,output_layer)
AEmodel.compile(loss='mse', optimizer=optimizer)
AEmodel.summary()

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=30,
                               verbose=1, 
                               mode='auto')

checkpoint1 = ModelCheckpoint('best_val_loss.h5',
                             monitor='val_loss',
                             save_best_only=True)

checkpoint2 = ModelCheckpoint('best_loss.h5',
                             monitor='loss',
                             save_best_only=True)

history = AEmodel.fit(processed_train, processed_target,
                      batch_size=16,
                      epochs=300,
#                       validation_split=0.2,
                      callbacks=[checkpoint2])
#                                      validation_data=(val, target_val))


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc='upper left')
plt.show()

AEmodel.save('AutoEncoderModelFull.h5')

# full_model_preds = AEmodel.predict(test)
full_train_preds = AEmodel.predict(full_train)

AEmodel.load_weights('best_loss.h5')
AEmodel.compile(loss='mse', optimizer=optimizer)
# preds = AEmodel.predict(test)
train_preds = AEmodel.predict(full_train)

AEmodel.evaluate(full_train, full_target)

AEmodel.save('AutoEncoderModelBestLoss.h5')

bsb = img.imread('https://github.com/sampath9dasari/GSU/raw/master/denoise_test.png')
# test = img.imread('../kaggle/working/test/1.png')
plt.imshow(bsb, cmap=plt.cm.gray)

# ii = cv2.imread("https://github.com/sampath9dasari/GSU/raw/master/denoise_test.png")
gray_image = cv2.cvtColor(bsb, cv2.COLOR_BGR2GRAY)
# print(gray_image)
plt.imshow(gray_image,cmap=plt.cm.gray)
plt.show()

gpred = AEmodel.predict(gray_image.reshape(1,1599,1200,1))

fig, ax = plt.subplots(1,2,figsize=(22,12))
ax[0].imshow(gray_image, cmap=plt.cm.gray)
ax[1].imshow(gpred.reshape(1600,1200), cmap=plt.cm.gray)

fig, ax = plt.subplots(3,2,figsize=(22,16))
ax[0][0].imshow(full_train[42].reshape(target_height,target_width), cmap=plt.cm.gray)
ax[0][1].imshow(full_target[42].reshape(target_height,target_width), cmap=plt.cm.gray)
ax[1][0].imshow(full_train_preds[42].reshape(target_height,target_width), cmap=plt.cm.gray)
ax[1][1].imshow(train_preds[42].reshape(target_height,target_width), cmap=plt.cm.gray)
reshape = cv2.resize(full_train_preds[42],(target_width,258))
ax[2][0].imshow(reshape.reshape(258,target_width), cmap=plt.cm.gray)
reshape = cv2.resize(train_preds[42],(target_width,258))
ax[2][1].imshow(reshape.reshape(258,target_width), cmap=plt.cm.gray)

# time
# ids = []
# vals = []
# file_list = glob.glob('/kaggle/working/test/*.png')
# file_list.sort()
# for i, f in enumerate(file_list):
#     file = os.path.basename(f)
#     imgid = int(file[:-4])
#     test_img = cv2.imread(f, 0)
#     img_shape = test_img.shape
# #     print('processing: {}'.format(imgid))
# #     print(img_shape)
#     preds_reshaped = cv2.resize(preds[i], (img_shape[1], img_shape[0]))
#     for r in range(img_shape[0]):
#         for c in range(img_shape[1]):
#             ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))
#             vals.append(preds_reshaped[r, c])

# print('Writing to csv file')
# pd.DataFrame({'id': ids, 'value': vals}).to_csv('submission.csv', index=False)

#Load and Scale test images into one big list.
file_list = glob.glob('/kaggle/working/test/*.png')
file_list.sort()
test_size = len(file_list)

#initailize data arrays.
img_ids = []
test = []

#read data
for i, img_dir in enumerate(file_list):
    file = os.path.basename(img_dir)
    imgid = int(file[:-4])
    img_ids.append(imgid)
    img_pixels = image.load_img(img_dir, color_mode='grayscale')
    w, h = img_pixels.size
    test.append(np.array(img_pixels).reshape(1, h, w, 1) / 255.)
    
print('Test sample shape: ', test[0].shape)
print('Test sample dtype: ', test[0].dtype)

#Predict test images one by one and store them into a list.
test_preds = []
for img in test:
    test_preds.append(AEmodel.predict(img)[0, :, :, 0])
    

fig, ax = plt.subplots(1,2,figsize=(22,12))
ax[0].imshow(test[45].reshape(test[45].shape[1],test[45].shape[2]), cmap=plt.cm.gray)
ax[1].imshow(test_preds[45].reshape(test[45].shape[1],test[45].shape[2]), cmap=plt.cm.gray)

fig, ax = plt.subplots(1,2,figsize=(16,8))
ax[0].imshow(test[42].reshape(test[42].shape[1],test[42].shape[2]), cmap=plt.cm.gray)
ax[1].imshow(test_preds[42].reshape(test[42].shape[1],test[42].shape[2]), cmap=plt.cm.gray)

# First column will be raw data, second column will be the corresponding cleaned images.
f, ax = plt.subplots(2,3, figsize=(20,10))
f.subplots_adjust(hspace = .1, wspace=.05)
for i, (img, lbl) in enumerate(zip(test[:3], test_preds[:3])):
    ax[0, i].imshow(img[0,:,:,0], cmap='gray')
    ax[0, i].title.set_text('Original Image')
    ax[0, i].axis('off')

    ax[1, i].imshow(lbl, cmap='gray')
    ax[1, i].title.set_text('Cleaned Image')
    ax[1, i].axis('off')
plt.show()

#Flatten the 'test_preds' list into 1-d list for submission.
submit_vector = []
submit_ids = []
for imgid, img in zip(img_ids,test_preds):
    h, w = img.shape
    for c in range(w):
        for r in range(h):
            submit_ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))
            submit_vector.append(img[r,c])
print(len(submit_vector))

len(submit_vector)

sample_csv = pd.read_csv('/kaggle/working/sampleSubmission.csv')
sample_csv.head(10)

id_col = sample_csv['id']
value_col = pd.Series(submit_vector, name='value')
submission = pd.concat([id_col, value_col], axis=1)
submission.head(10)

submission.to_csv('submission.csv',index = False)

shutil.rmtree('/kaggle/working/train')
shutil.rmtree('/kaggle/working/test')
shutil.rmtree('/kaggle/working/train_cleaned')


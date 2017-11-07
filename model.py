import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

samples = []
with open('./data/Training_Data/driving_log.csv') as csvfile:
   reader=csv.reader(csvfile)
   for line in reader:
     samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def augment_brightness(image):
	"""
	apply random brightness on the image
	"""
	image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	random_bright = .25+np.random.uniform()
	
	# scaling up or down the V channel of HSV
	image[:,:,2] = image[:,:,2]*random_bright
	return image

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def generator(samples,batch_size=32):
    num_samples=len(samples)
    while 1:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            steering_correction = [0.0, 0.17, -0.17]
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('\\')[-1]
                    current_path='./data/Training_Data/IMG/'+filename
                    image = cv2.imread(current_path)

                    #Adding center camera, left camera and right camera images with steering correction
                    images.append(image)
                    measurement = float(batch_sample[3]) + steering_correction[i]
                    measurements.append(measurement)

                    #Augmenting center, left and right camera images with some random brightness
                    image_aug_brght = augment_brightness(image)
                    images.append(image_aug_brght)
                    measurement_aug_brght = float(batch_sample[3]) + steering_correction[i]
                    measurements.append(measurement_aug_brght)

                    #Augmenting center, left and right camera images with some random shadow
                    image_rnd_shd = add_random_shadow(image)
                    measurement_rnd_shd = float(batch_sample[3]) + steering_correction[i]
                    images.append(image_rnd_shd)
                    measurements.append(measurement_rnd_shd)

                    #Augmenting center, left and right camera images by flipping them and multiplying steering measurement by -1.0
                    image_flip = cv2.flip(image,1)
                    measurement_flip = (float(batch_sample[3]) + steering_correction[i])*-1.0
                    images.append(image_flip)
                    measurements.append(measurement_flip)
                    
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
         
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
 
model=Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*12, validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*12, nb_epoch=4)
model.save('model.h5') 

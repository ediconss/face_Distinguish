import numpy as np
import cv2
import os

images=[]
labels=[]
def load_images(file_path):
    for file in os.listdir(file_path):
        full_path = os.path.abspath(os.path.join(file_path, file))
        if os.path.isdir(full_path):
            for filea in os.listdir(full_path):
                print(filea)
                if filea == 'my':
                    for i in os.listdir(full_path+'/'+filea):
                        image=cv2.imread(full_path+'/'+filea+'/'+i)
                        images.append(image)
                        labels.append(1)
                if filea == 'other':
                    for i in os.listdir(full_path+'/'+filea):
                        image=cv2.imread(full_path + '/' + filea + '/' + i)
                        images.append(image)
                        labels.append(0)
    return images,labels
def load_data(file_path):
    images, labels=load_images(file_path)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels
if __name__== '__main__':
    images, labels=load_data(r'D:\python\MyTest')
    print(images.shape[1:4])
    print(labels[234])
    print(images.shape)
    cv2.imshow("test",images[234])
    cv2.waitKey(0)
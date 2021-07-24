# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2 as cv
import os
import numpy as np
import pandas as pd
import csv

os.system('clear')

vids_path   = "/home/ibrahim/Projects/Datasets/HPO_Recording/Videos/"
images_path = "/home/ibrahim/Projects/Datasets/HPO_Recording/Images/"

os.chdir(vids_path)

f = open(images_path + 'Annotations.csv', 'a', newline='')
writer = csv.writer(f)

video_names =  sorted(os.listdir(vids_path), key = lambda x: int(x[0]))
# print(video_names)

for i in range(len(video_names)):
    print(video_names[i][0])

    cap = cv.VideoCapture(vids_path + video_names[i])
    frames = 0
    name = video_names[i][0]+'_image_'
    print(name)
    lookUpTable = np.empty((1,256), np.uint8)

    while cap.isOpened():
        frames += 1
        ret, frame = cap.read()

        if ret == True:
            gamma     = round(np.random.uniform(0,2),3)

            # LookUp table for improving compute time
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
                
            gray      = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            procFrame = cv.LUT(gray, lookUpTable)

            if (frames%15 == 0):
                cv.imwrite(images_path + name + str(frames) + ".jpg", procFrame)
                writer.writerow([name + str(frames) + ".jpg", gamma])
                

            cv.imshow('frame', gray)
            cv.imshow('proc frame', procFrame)

            size = gray.shape
            if cv.waitKey(1) == ord('q'):
                break
        else: 
            break
        
    cap.release()

f.close()
cv.destroyAllWindows()







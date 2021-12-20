# -*- coding: utf-8 -*-  

"""
Created on 2021/05/11

@author: Ruoyu Chen
"""

import os
import random

from tqdm import tqdm

def main():
    datasets_dir = "/home/cry/data2/VGGFace2/train_align_arcface"

    # Get the people lists
    peoples = os.listdir(datasets_dir)
    peoples.sort()

    people_num = 0
    for people in tqdm(peoples):
        people_images = os.listdir(os.path.join(datasets_dir,people))

        random.shuffle(people_images)

        for people_image in people_images[:int(len(people_images)*0.8)]:
            with open("./train.txt","a") as file:
                doc = os.path.join(people,people_image)+" "+str(people_num)+"\n"
                file.write(doc)

        for people_image in people_images[int(len(people_images)*0.8):int(len(people_images)*0.9)]:
            with open("./val.txt","a") as file:
                doc = os.path.join(people,people_image)+" "+str(people_num)+"\n"
                file.write(doc)

        for people_image in people_images[int(len(people_images)*0.9):]:
            with open("./test.txt","a") as file:
                doc = os.path.join(people,people_image)+" "+str(people_num)+"\n"
                file.write(doc)

        people_num += 1       

if __name__ == "__main__":
    main()


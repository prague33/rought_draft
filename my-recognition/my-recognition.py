#! /usr/bin/python3

import jetson.inference
import jetson.utils

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="model to use")
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--labels", type=str, help="labels to use")
opt = parser.parse_args()

img = jetson.utils.loadImage(opt.filename)


## make up parameters for 'imageNet' to recognize our custom model
imageNetArgs = [ '', f"--model={opt.model}", f"--labels={opt.labels}", 
        "--input_blob=input_0", "--output_blob=output_0", opt.filename ]

# network parameter ('googlenet') is not really used as we'll use our custom model
net = jetson.inference.imageNet("googlenet", imageNetArgs)

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)

print(f"image is recognized as '{class_desc}' (class {class_idx}) with  confidence level {confidence*100}")



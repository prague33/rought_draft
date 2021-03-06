# Home Products: Safe or Dangerous for Little Kids

This project tries to train an AI model that tells whether a home product
is safe or dangerous for little kids to touch or play with. The model takes
a home product package photo and tries to infer whether it should be labeled
as "safe" or "danger".

All the data in this repo is generated by the author. Due to limited time, the
data set is relatively small.

## Running the model for inference

Suppose you're in the subdirectory `training`, you can run the following command
to run the inference code:

```
python3 ../my-recognition/my-recognition.py \
	--labels=data/homeproducts/labels.txt \
	models/homeproducts/resnet18.onnx \
	inputfilename
```

where the parameter `models/homeproducts/resnet18.onnx` can be replaced with your custom model, and 
`inputfilename` can be any image file.

## Re-train the model

I have pre-trained the model based on `resnet18`. The model is located at directory
`training/models/homeproducts` - see files `resnet18.onnx.*`.

I used the script `train.py` from `jetson-inference` to train the `resnet18` model. The command used is:

```
python3 train.py --model-dir=models/homeproducts data/homeproducts
```

when I run this command, I am at directory `jetson-inference/python/training/classification`, and 
symbol links for `models/homeproducts` and `data/homeproducts` are setup to the right directories.

If anyone wants to re-train the model, you can try changing the parameters to `train.py` to see how it goes.



# How To Use David Sandberg's Facenet Implementation

This guide describes how to employ [David Sandberg's Facenet Implementation](https://github.com/davidsandberg/facenet/) to train your own face classifier.

## 1. Installation/Dependencies

```shell
# Install Tensorflow using pip
pip install tensorflow

# Clone David Sandberg's facenet repo.
git clone https://github.com/davidsandberg/facenet.git

# Set PYTHONPATH
export PYTHONPATH=/path/to/facenet/src

# ... or use https://direnv.net/ for setting PYTHONPATH;
# Direnv loads env variables from .envrc, whenever entering a directory.
#
# edit /path/to/facenet/.envrc:
export PYTHONPATH=$(pwd)/src
# don't forget to allow updated .envrc
direnv allow
```

## 2. Face Alignment

From https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1.

In order to train a classifier on your own images, a training set must be created;
this requires aligning faces, i.e. automatically cropping all parts of an image containing a face.

All images files must be structured as follows.

```
/path/to/facenet/images
  raw
    person1
      person1_001.jpg
    person2
      person2_001.jpg
      person2_002.jpg
      person2_003.jpg
    person3
      person3_001.jpg
      person3_002.jpg
    :
```

Now, align faces using the following command.

```shell
cd /path/to/facenet
python src/align/align_dataset_mtcnn.py \
  images/raw/ \
  images/train_mtcnnpy_182/ \
  --image_size 182 \
  --margin 44
```

_(You certainly set `PYTHONPATH`, didn't you? See section above.)_

Or, in parallel, using 4 processes.

```shell
for N in {1..4}; do \
python src/align/align_dataset_mtcnn.py \
  images/raw/ \
  images/train_mtcnnpy_182/ \
  --image_size 182 \
  --margin 44 \
  --random_order \
  --gpu_memory_fraction 0.25 \
& done
```

This will detect the largest face in each image. To detect multiple faces in an image, add the `--detect_multiple_faces` parameter as follows.

```shell
cd /path/to/facenet
python src/align/align_dataset_mtcnn.py \
  ~/images/raw
  ~/images/train_mtcnnpy_182 \
  --image_size 182 \
  --margin 44 \
  --detect_multiple_faces 1
```

Or, detect multiple faces using all available CPUs (requires `bc`, which should come with Linux by default).

```shell
for N in {1..$(nproc)}
do 
  python src/align/align_dataset_mtcnn.py \
  images/raw/ 
  images/train_mtcnnpy_182 \
  --image_size 182 \
  --margin 44 \
  --random_order \
  --gpu_memory_fraction $(echo "scale=2;1/$(nproc)"|bc) \
  --detect_multiple_faces 1 \
& done
```

### Selecting Relevant Images

For each face detected, a file named `~/images/train_mtcnnpy_182/NAME_N.png` will be written. Since all faces in the image have been detected, some faces may not belong to the person wanted. To select the faces of the person sought only, `feh` can be used.

First, move all images extracted from `~/images/train_mtcnnpy_182/` to `~/images/train_mtcnnpy_182/ALL`. Then use `feh` to copy relevant images one directory up, from `~/images/train_mtcnnpy_182/ALL` to `~/images/train_mtcnnpy_182/`. Now, `~/images/train_mtcnnpy_182/` contains only relevant images.

```shell
cd ~/images/train_mtcnnpy_182/
mkdir ALL
mv *.png ALL
cd ALL
feh -A "cp %f .."
# Now cycle through all images using the arrow keys;
# to select an image (i.e. copy one dir up), press '0';
# press 'q' to quit feh;
# now, all images selected are in ~/images/train_mtcnnpy_182/.
```

## 3. Training Your Own Classifier

From https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images.

After aligning faces, you can train you own classifier, which requires a model.

Training your own model would take weeks, thus, download a pre-trained model.
The model will create a unique *fingerprint* for each face;
fingerprints of the same face are very close; fingerprints of different faces are far apart.
Fingerprint will be used by the classifier trained here, to classify faces, i.e. tell which faces belong to the same person.

The latest pre-trained models can be found [here](https://github.com/davidsandberg/facenet#pre-trained-models).

```shell
mkdir /path/to/facenet/models/
# Download and extract https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55- into /path/to/facenet/models/
# You should now have /path/to/facenet/models/20180402-114759/.
```

Training requires a training set, which is located at `images/train_mtcnnpy_182`. Validating the model after training requires a test set. Extract some images (~ 10 %) from `images/train_mtcnnpy_182` into `images/test_mtcnnpy_182`.

```shell
cd images
mkdir -p test_mtcnnpy_182/bandera
mkdir -p test_mtcnnpy_182/hitler
find train_mtcnnpy_182/bandera/ -maxdepth 1 -name '*.png' | shuf | head -10
find train_mtcnnpy_182/hitler/ -maxdepth 1 -name '*.png' | shuf | head -10
```

Now train.

```shell
cd /path/to/facenet/
python src/classifier.py TRAIN \
  images/train_mtcnnpy_182/ \
  models/20180402-114759/20180402-114759.pb \
  models/classifier.pkl \
  --batch_size 1000
```

## 4. Use the Classifier

Finally, you can use your own classifier to classify faces.

This assumes that you have prepared a set of test/validation images,
in the same way as described above, under *Face Alignment*.
The aligned faces you wish to recognize are stored in `images/test_mtcnnpy_182`.

```shell
cd /path/to/facenet/
python src/classifier.py CLASSIFY \
  images/test_mtcnnpy_182/ \
  models/20180402-114759/20180402-114759.pb \
  models/classifier.pkl \
  --batch_size 1000
```

This will produce output similar to the following.

```
Number of classes: 10
Number of images: 50
Loading feature extraction model
Model filename: /path/to/facenet/models/20180402-114759/20180402-114759.pb
Calculating features for images
Testing classifier
Loaded classifier model from file "/path/to/facenet/models/classifier.pkl"
   0  Ariel Sharon: 0.452
   1  Ariel Sharon: 0.376
   2  Ariel Sharon: 0.426
...
...
...
  47  Vladimir Putin: 0.418
  48  Vladimir Putin: 0.453
  49  Vladimir Putin: 0.378
Accuracy: 1.000
```

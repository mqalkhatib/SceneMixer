import os, glob, random
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import *


dataset = "EuroSAT_RGB"    # root with class subfolders

if dataset == "UC":
    DATA_DIR = "UCMerced_LandUse"
    RAW_DECODE_SIZE = (256, 256)
    
elif dataset == "AID":
    DATA_DIR = "AID"
    RAW_DECODE_SIZE = (600, 600)
    
elif dataset == "NWPU":
    DATA_DIR = "NWPU-RESISC45"
    RAW_DECODE_SIZE = (256, 256)
         
else:
    DATA_DIR = 'EuroSAT_RGB'
    RAW_DECODE_SIZE = (64, 64)


BATCH    = 32
SEED     = 2 #1337
AUTOTUNE = tf.data.AUTOTUNE

#random.seed(SEED)

# ---------- enumerate files per class ----------
class_names = sorted([d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d))])

def list_images(cls):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(DATA_DIR, cls, e)))
    return sorted(paths)

per_class = {c: list_images(c) for c in class_names}
NUM_CLASSES = len(class_names)

# ---------- stratified split 70/15/15, no overlap ----------
train_files, val_files, test_files = [], [], []
train_labels, val_labels, test_labels = [], [], []

for label, cls in enumerate(class_names):
    files = per_class[cls]
    # deterministic per-class shuffle
    rnd = random.Random(SEED)
    rnd.shuffle(files)

    n = len(files)
    n_train = int(round(0.70 * n))
    n_val   = int(round(0.15 * n))
    n_test  = n - n_train - n_val
    if n_test < 0:  # rare rounding fix
        n_val += n_test
        n_test = 0

    tr = files[:n_train]
    va = files[n_train:n_train+n_val]
    te = files[n_train+n_val:]

    train_files += tr; train_labels += [label]*len(tr)
    val_files   += va; val_labels   += [label]*len(va)
    test_files  += te; test_labels  += [label]*len(te)

# safety: ensure no overlap
def _assert_disjoint(a, b, c):
    sa, sb, sc = set(a), set(b), set(c)
    assert sa.isdisjoint(sb) and sa.isdisjoint(sc) and sb.isdisjoint(sc), "Overlap detected!"
_assert_disjoint(train_files, val_files, test_files)

print(f"Classes: {NUM_CLASSES}")
print(f"Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")


def _read_any_image_py(path_bytes):
    p = path_bytes.decode("utf-8")
    with Image.open(p) as im:
        im = im.convert("RGB")               # handles TIFF/PNG/JPEG/… → RGB
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def make_ds(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        img = tf.numpy_function(_read_any_image_py, [path], tf.float32)
        img.set_shape([None, None, 3])       # important for shape inference
        img = tf.image.resize(img, RAW_DECODE_SIZE)
        y   = tf.one_hot(label, depth=NUM_CLASSES)
        return img, y

    if shuffle:
        ds = ds.shuffle(buffer_size=min(5000, len(paths)), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
    # Strongly recommended to cache when decoding in Python:
    ds = ds.cache()                          # RAM cache; or .cache("ucm_cache_train")
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_files, train_labels, shuffle=True)
val_ds   = make_ds(val_files,   val_labels,   shuffle=False)
test_ds  = make_ds(test_files,  test_labels,  shuffle=False)



for idx, cls in enumerate(class_names):
    n_train = train_labels.count(idx)
    n_val   = val_labels.count(idx)
    n_test  = test_labels.count(idx)
    total   = n_train + n_val + n_test
    print(f"{cls:25s}  Train: {n_train:3d}   Val: {n_val:3d}   Test: {n_test:3d}   Total: {total:3d}")
print(f"\nTrain: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")

###############################################################################
# Building the Model
from tensorflow.keras import layers, models, activations

def SceneMixer(X, num_classes, patch=4, dim=256, depth=8, target_size = 64):
    inputs = layers.Input(shape=(None, None, 3))   # flexible input size
    x = layers.Resizing(target_size, target_size)(inputs)
    #inputs = layers.Input(shape=X.shape[1:])  # (H, W, C)
    # Patch embedding
    x = layers.Conv2D(dim, patch, strides=patch, padding="valid")(x)
    x = activations.gelu(x); 
    x = layers.BatchNormalization()(x)
    # Repeated ConvMixer blocks
    for _ in range(depth):
        y1 = layers.DepthwiseConv2D(3, padding="same")(x)
        y1 =  activations.gelu(y1); 
        y1 = layers.BatchNormalization()(y1)
        
        
        y2 = layers.DepthwiseConv2D(5, padding="same")(x)
        y2 =  activations.gelu(y2); 
        y2 = layers.BatchNormalization()(y2)        
        
        y = layers.Add()([y1, y2])  
        x = layers.Add()([x, y])                    # residual
        
        x = layers.Conv2D(dim, 1, padding="same")(x)
        x = activations.gelu(x); 
        x = layers.BatchNormalization()(x)
        
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model 


model =   SceneMixer(None, num_classes=NUM_CLASSES, patch=4, dim=128, depth=4, target_size = 64)

model.summary()
from net_flops import net_flops
net_flops(model)

# Define a callback to modify the learning rate dynamically
import keras
checkpoint = keras.callbacks.ModelCheckpoint(
        f"Model_Weights/{dataset}_{SceneMixer}_{SEED}.h5",
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=10,
        min_lr=5e-5
        )
    

history = model.fit(train_ds, 
                    validation_data=val_ds, 
                    epochs=100, 
                    callbacks=[checkpoint, lr_callback]
                    #callbacks=[]
                    )

display_history(history)


model.load_weights(f"Model_Weights/{dataset}_{SceneMixer}_{SEED}.h5")


show_random_test_prediction(model, test_ds, class_names)


oa, aa, kappa, cm, each_acc = evaluate_model(model, test_ds, class_names)


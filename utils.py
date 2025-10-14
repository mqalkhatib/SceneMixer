import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import os, glob, random
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score





def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def display_history(history):
    # Retrieve loss and accuracy data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # Create a figure with 2 horizontal subplots
    plt.figure(figsize=(12, 5))
    
    # Subplot for training and validation loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot for training and validation accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    # Show the combined figure
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    
    # Get training history
    val_acc = history.history['val_accuracy']
    best_epoch = np.argmax(val_acc)
    best_val = val_acc[best_epoch]

    
    # Plot Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], 'y', label='Train Acc')
    plt.plot(val_acc, 'r', label='Val Acc')
    
    # Mark best epoch
    plt.axvline(best_epoch, color='k', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.scatter(best_epoch, best_val, color='black')
    plt.text(best_epoch, best_val, f"{best_val:.2f}", fontsize=10, color='black', va='bottom')
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
     

def show_random_test_prediction(model, test_ds, class_names, shuffle_buf=10000):
    # flatten to individual examples, shuffle every call, then take 1
    for img, label in test_ds.unbatch().shuffle(shuffle_buf, reshuffle_each_iteration=True).take(1):
        img = img.numpy()
        # handle one-hot or integer labels
        if tf.rank(label) == 0:
            true_id = int(label.numpy())
        else:
            true_id = int(tf.argmax(label).numpy())
        true_name = class_names[true_id]

        # predict
        logits = model.predict(img[None, ...], verbose=0)
        pred_id = int(np.argmax(logits[0]))
        pred_name = class_names[pred_id]
        conf = float(np.max(logits[0]))

        # visualize
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.axis("off")
        color = "green" if pred_id == true_id else "red"
        plt.title(f"Pred: {pred_name} (p={conf:.3f})\nTrue: {true_name}", color=color, fontsize=12)
        plt.show()



def evaluate_model(model, test_ds, class_names):
    # Collect predictions and true labels
    y_true, y_pred = [], []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0)       # (batch, num_classes)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(batch_y.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall Accuracy
    oa = accuracy_score(y_true, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False)
    plt.gca().set_aspect('equal', adjustable='box')  # keep cells square
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    # Average Accuracy (per-class accuracies → mean)
    each_acc, aa = AA_andEachClassAccuracy(cm)
    #class_acc = cm.diagonal() / cm.sum(axis=1)
    #aa = np.mean(class_acc)

    # Cohen’s Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Print
    print(f"✅ Overall Accuracy (OA): {oa*100:.2f}%")
    print(f"📊 Average Accuracy (AA): {aa*100:.2f}%")
    print(f"📈 Kappa Score: {kappa*100:.2f}")

    return oa, aa, kappa, cm, each_acc
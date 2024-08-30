import gc
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(1000)
tf.random.set_seed(1000)

# Define image shape and batch size
image_shape = (224, 224, 3)
BATCH_SIZE = 32
NUM_FOLDS = 5

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Load datasets
train_ds = keras.utils.image_dataset_from_directory(
    directory='D:/cyber security/TRAIN', 
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(224, 224)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='D:/cyber security/VALIDATION', 
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(224, 224)
)

test_ds = keras.utils.image_dataset_from_directory(
    directory='D:/cyber security/TEST', 
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(224, 224)
)

# Function to apply data augmentation to each image in the batch
def augment_data(images, labels):
    def augment(image):
        image = tf.image.convert_image_dtype(image, tf.float32)  # Ensure image dtype is float32
        image = tf.numpy_function(datagen.random_transform, [image], tf.float32)
        return image

    images = tf.map_fn(augment, images, dtype=tf.float32)
    return images, labels

# Apply data augmentation to the training dataset
train_ds = train_ds.map(augment_data)

# Load VGG16 model pre-trained on ImageNet, excluding the top (classification) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
MODEL = Sequential([
    base_model,
    Flatten(),
    Dense(4096),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(4096),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1),
    Activation('sigmoid')
])

# Define custom F1 score metric
def f1_score(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_val

# Initialize K-Fold cross-validator
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=45)

# Unbatch the dataset for KFold splitting
train_ds_np = list(train_ds.unbatch().as_numpy_iterator())
images = np.array([item[0] for item in train_ds_np])
labels = np.array([item[1] for item in train_ds_np])

# Cross-validation
FOLD_NO = 1
HISTORIES = []
fold_accuracies = []

for train_index, val_index in kfold.split(train_ds_np):
    print(f'Training fold {FOLD_NO}...')

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    MODEL.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer, 
                  metrics=[
                      keras.metrics.BinaryAccuracy(name='accuracy'),
                      keras.metrics.Precision(name='precision'),
                      keras.metrics.Recall(name='recall'),
                      f1_score
                  ])
    
    # Split dataset into training and validation sets
    train_images, val_images = images[train_index], images[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]
    
    train_fold = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(BATCH_SIZE)
    val_fold = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(BATCH_SIZE)

    # Train the model
    HIST = MODEL.fit(train_fold, validation_data=val_fold, epochs=50, batch_size=BATCH_SIZE, callbacks=[lr_reduction, early_stopping])
    HISTORIES.append(HIST)
    
    # Store the validation accuracy for this fold
    val_accuracy = HIST.history['val_accuracy'][-1]
    fold_accuracies.append(val_accuracy)
    
    FOLD_NO += 1
    
    # Free up memory
    del train_fold, val_fold, train_images, val_images, train_labels, val_labels
    gc.collect()

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Fine-tuning with a lower learning rate
fine_tune_optimizer = keras.optimizers.Adam(learning_rate=0.00001)

MODEL.compile(loss=keras.losses.binary_crossentropy, optimizer=fine_tune_optimizer, 
              metrics=[
                  keras.metrics.BinaryAccuracy(name='accuracy'),
                  keras.metrics.Precision(name='precision'),
                  keras.metrics.Recall(name='recall'),
                  f1_score
              ])

# Fine-tune the model
fine_tune_hist = MODEL.fit(train_ds, validation_data=validation_ds, epochs=30, batch_size=BATCH_SIZE, callbacks=[lr_reduction, early_stopping])
HISTORIES.append(fine_tune_hist)

# Function to plot metrics
def plot_metrics(histories):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for i, history in enumerate(histories):
            plt.plot(history.history[metric], label=f'Train {metric} - Fold {i+1}')
            plt.plot(history.history[f'val_{metric}'], label=f'Val {metric} - Fold {i+1}')
        
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.show()

# Plot the metrics
plot_metrics(HISTORIES)

# Evaluate on the test dataset
print('Evaluating on the test dataset...')
results = MODEL.evaluate(test_ds)
for name, value in zip(MODEL.metrics_names, results):
    print(f"{name}: {value}")

# Print average accuracy across folds
average_accuracy = np.mean(fold_accuracies)
print(f'Average accuracy across all folds: {average_accuracy:.4f}')

# Predict on the test set
print('Generating predictions on the test set...')
Y_pred = MODEL.predict(test_ds)
y_Pred = np.round(Y_pred).astype(int).flatten()
y_True = np.concatenate([y for x, y in test_ds], axis=0)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_True, y_Pred)
class_report = classification_report(y_True, y_Pred, target_names=['Class 0', 'Class 1'])

print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

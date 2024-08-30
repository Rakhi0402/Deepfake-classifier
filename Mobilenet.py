import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Directories
test_dir = r'C:\Users\pooja\OneDrive\Documents\AI VS CAMERA\TEST'
train_dir = r'C:\Users\pooja\OneDrive\Documents\AI VS CAMERA\TRAIN'
validation_dir = r'C:\Users\pooja\OneDrive\Documents\AI VS CAMERA\VALIDATION'

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Parameters
batch_size = 32
epochs = 20
num_folds = 5

# Load and preprocess the datasets
def get_file_list_and_labels(directory):
    classes = os.listdir(directory)
    file_list = []
    labels = []
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        for file_name in os.listdir(class_dir):
            file_list.append(os.path.join(class_dir, file_name))
            labels.append(idx)
    return file_list, labels

file_list, labels = get_file_list_and_labels(train_dir)
file_list = np.array(file_list)
labels = np.array(labels)

# Define K-fold Cross Validator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# K-fold Cross Validation model evaluation
acc_per_fold = []
loss_per_fold = []
fold_no = 1

for train_index, val_index in kf.split(file_list):
    print(f'Training fold {fold_no}...')
    
    train_files, val_files = file_list[train_index], file_list[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]
    
    # Convert labels to strings
    train_labels = [str(label) for label in train_labels]
    val_labels = [str(label) for label in val_labels]
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_files, 'class': train_labels}),
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': val_files, 'class': val_labels}),
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Define the MobileNet model with custom top layers
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze all layers of MobileNet
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

    # Evaluate the model
    scores = model.evaluate(validation_generator, steps=len(validation_generator))
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    fold_no += 1

# Average metrics across all folds
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(len(acc_per_fold)):
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

# Evaluate on the test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_accuracy:.2f}')

# Predict classes on the test set
test_generator.reset()  # Ensure the generator starts from the beginning
Y_pred = model.predict(test_generator, steps=np.ceil(test_generator.samples / test_generator.batch_size))
y_pred = np.round(Y_pred).astype(int)

# Calculate confusion matrix and classification report
y_true = test_generator.classes
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=['AI', 'camera'], output_dict=True)

precision = class_report['weighted avg']['precision']
recall = class_report['weighted avg']['recall']
f1_score = class_report['weighted avg']['f1-score']
accuracy = test_accuracy

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')
print(f'Accuracy: {accuracy:.2f}')

# Plot training & validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

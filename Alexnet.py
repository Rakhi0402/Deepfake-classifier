from tensorflow.keras.applications import VGG16
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Directories
test_dir = r'D:\cyber security\TEST'
train_dir = r'D:\cyber security\TRAIN'
validation_dir = r'D:\cyber security\VALIDATION'

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
image_shape = (224, 224, 3)
learning_rate = 0.001

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

# Define the model
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
    for layer in base_model.layers:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), 
        Dense(4096),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(4096),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')  # Sigmoid for binary classification
    ])
    
    return model

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
    
    train_df = pd.DataFrame({'filename': train_files, 'class': train_labels})
    val_df = pd.DataFrame({'filename': val_files, 'class': val_labels})
    
    # Flow from dataframe generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Create model
    model = create_model()
  
    # Compile the model with binary crossentropy
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # Calculate steps_per_epoch and validation_steps
    steps_per_epoch = len(train_files) // batch_size
    validation_steps = len(val_files) // batch_size

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    # Evaluate the model on validation data
    scores = model.evaluate(validation_generator, steps=validation_steps)
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
print(f'Test accuracy: {test_accuracy:.4f}')

# Predict classes on the test set
test_generator.reset()  # Ensure the generator starts from the beginning
Y_pred = model.predict(test_generator, steps=int(np.ceil(test_generator.samples / test_generator.batch_size)))
y_pred = np.round(Y_pred).astype(int)

# Calculate confusion matrix and classification report
y_true = test_generator.classes
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=['AI', 'Camera'], output_dict=True)

precision = class_report['weighted avg']['precision']
recall = class_report['weighted avg']['recall']
f1_score = class_report['weighted avg']['f1-score']
accuracy = test_accuracy

print('Classification Report:')
for label, metrics in class_report.items():
    if isinstance(metrics, dict):  # Skip 'accuracy' as it is just a single float value
        print(f'\nClass: {label}')
        for metric, value in metrics.items():
            print(f'{metric.capitalize()}: {value:.4f}')
    else:
        print(f'\nOverall {label.capitalize()}: {metrics:.4f}')
        
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Visualize confusion matrix
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['AI', 'camera'], rotation=45)
plt.yticks(tick_marks, ['AI', 'camera'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

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

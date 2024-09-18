import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

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

class GeneticAlgorithm:
    def __init__(self, population_size, n_generations, mutation_rate, crossover_rate, learning_rate_range=[1e-5, 1e-2]):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.learning_rate_range = learning_rate_range

    # Initialize the population with random learning rates within the defined range
    def initialize_population(self):
        return [random.uniform(self.learning_rate_range[0], self.learning_rate_range[1]) for _ in range(self.population_size)]

    # Fitness function: Evaluate the accuracy of a learning rate by training the model on a single epoch
    def fitness(self, learning_rate, model, train_generator, validation_generator):
        # Compile the model with the current learning rate
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model for a single epoch and get the validation accuracy
        history = model.fit(train_generator, epochs=1, validation_data=validation_generator, verbose=0)
        val_accuracy = history.history['val_accuracy'][-1]
        
        return val_accuracy

    # Selection function: Roulette wheel selection based on fitness scores
    def selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness_score in enumerate(fitness_scores):
            current += fitness_score
            if current > pick:
                return population[i]

    # Crossover function: Combine two parents' learning rates
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            return (parent1 + parent2) / 2
        return parent1

    # Mutation function: Mutate the learning rate by a small random factor
    def mutation(self, offspring):
        if random.random() < self.mutation_rate:
            mutation_factor = random.uniform(0.9, 1.1)
            offspring *= mutation_factor
            # Ensure offspring stays within the learning rate range
            offspring = max(self.learning_rate_range[0], min(offspring, self.learning_rate_range[1]))
        return offspring

    # Main optimization loop: Evolve the population over several generations
    def optimize(self, model, train_generator, validation_generator):
        # Step 1: Initialize the population
        population = self.initialize_population()
        best_solution = None
        best_fitness = -1

        # Step 2: Iterate through generations
        for generation in range(self.n_generations):
            # Evaluate fitness for each learning rate in the population
            fitness_scores = [self.fitness(lr, model, train_generator, validation_generator) for lr in population]

            # Track the best solution found so far
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_solution = population[fitness_scores.index(max_fitness)]

            print(f"Generation {generation+1}/{self.n_generations}, Best learning rate: {best_solution}, Best fitness: {best_fitness}")

            # Step 3: Selection and crossover to generate new offspring
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.selection(population, fitness_scores)
                parent2 = self.selection(population, fitness_scores)
                offspring1 = self.crossover(parent1, parent2)
                offspring2 = self.crossover(parent1, parent2)
                
                # Apply mutation
                offspring1 = self.mutation(offspring1)
                offspring2 = self.mutation(offspring2)
                
                new_population += [offspring1, offspring2]

            # Replace the old population with the new one
            population = new_population

        return best_solution
    
# K-fold Cross Validation model evaluation
acc_per_fold = []
loss_per_fold = []
fold_no = 1

for train_index, val_index in kf.split(file_list):
    print(f'Training fold {fold_no}...')
    
    train_files, val_files = file_list[train_index], file_list[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]
    
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

    # Define the MobileNet model with Max pooling
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Adjust output units for 2 classes

    model = Model(inputs=base_model.input, outputs=predictions)

    # Print model summary to debug layer shapes
    print(model.summary())

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Define and use the Genetic Algorithm for learning rate optimization
    ga = GeneticAlgorithm(population_size=10, n_generations=3, mutation_rate=0.1, crossover_rate=0.5)
    best_learning_rate = ga.optimize(model, train_generator, validation_generator)
    print(f'Best learning rate: {best_learning_rate}')

    # Compile the model with the best learning rate found
    model.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Evaluate the model
    scores = model.evaluate(validation_generator)
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
Y_pred = model.predict(test_generator, steps=int(np.ceil(test_generator.samples / test_generator.batch_size)))
y_pred = np.round(Y_pred).astype(int)

# Calculate confusion matrix and classification report
y_true = test_generator.classes
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=['AI', 'camera'], output_dict=True)

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
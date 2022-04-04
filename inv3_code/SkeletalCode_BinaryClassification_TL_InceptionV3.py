from __future__ import print_function

import numpy
from itertools import chain
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, auc
matplotlib.use('Agg')

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          file_name='none'):

    accuracy = numpy.trace(cm) / float(numpy.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = numpy.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig(file_name + '.jpg')

def summarize_diagnostics(history, file_name="none"):
    # loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    # accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['binary_accuracy'], color='blue', label='train')
    plt.plot(history.history['val_binary_accuracy'], color='orange', label='test')
    # save plot to file
    plt.savefig(file_name + '.jpg')
    plt.close()

def define_classifier(base_model, nb_unit, nb_FClayer, dr_rate):
    nb_unit = 20 * nb_unit
    dr_rate = dr_rate * 0.1

    classifier = base_model.output
    classifier = GlobalAveragePooling2D()(classifier)

    if nb_FClayer == 0:
        classifier = Dropout(dr_rate)(classifier)
    elif nb_FClayer == 1:
        classifier = Dense(nb_unit, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
    elif nb_FClayer == 2:
        classifier = Dense(nb_unit, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = Dense(int(nb_unit / 2), activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
    else:
        classifier = Dense(nb_unit, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = Dense(nb_unit * 2, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = Dense(nb_unit, activation='relu', kernel_initializer='he_uniform')(classifier)
        classifier = Dropout(dr_rate)(classifier)
        classifier = BatchNormalization()(classifier)

    out = Dense(1, activation='sigmoid')(classifier)
    model = Model(inputs=base_model.inputs, outputs=out)

    return model

# Constants
nb_unit=3    #nb_unit = 20 * nb_unit
nb_FClayer=3
dr_rate=4    #dr_rate = dr_rate * 0.1
opt='adam'
tf_lrs = 0.001    #learning rate for fitting classifier
tune_lrs = 0.0001    #learning rate for fine tunning
decays = 0.01
bs = 32    #batch size
nb_epochs = 300

# Data Path
train_dir = '훈련데이터 경로 입력'    #eg: C:/Users/drbon/PycharmProjects/A4/Cancer_data/train'
validation_dir = '검증데이터 경로 입력'    #eg: 'C:/Users/drbon/PycharmProjects/A4/Cancer_data/validation'
test_dir = '테스트데이터 경로 입력'    #eg: 'C:/Users/drbon/PycharmProjects/A4/Cancer_data/test'

# Data Generator
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip='True',
                                       fill_mode='nearest')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_it = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), color_mode='rgb', class_mode='binary', batch_size=bs)
val_it = val_datagen.flow_from_directory(validation_dir, target_size=(224, 224), color_mode='rgb', class_mode='binary', batch_size=bs)
test_it = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), color_mode='rgb', class_mode='binary', batch_size=1)

train_filenames = train_it.filenames
nb_train_samples = len(train_filenames)
val_filenames = val_it.filenames
nb_val_samples = len(val_filenames)
test_filenames = test_it.filenames
nb_test_samples = len(test_filenames)

# Define Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freezing
for layer in base_model.layers:
    layer.trainable = False

model = define_classifier(base_model, nb_unit, nb_FClayer, dr_rate)

if opt == 'sgd':
    opti = SGD(lr=tf_lrs, decay=decays, momentum=0.9)
elif opt == 'adam':
    opti = Adam(lr=tf_lrs, decay=decays)

FN_score = tf.keras.metrics.FalseNegatives()
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['binary_accuracy', FN_score])

# Learning
es = EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=15, min_delta=0.001, restore_best_weights=True)
checkpoint = ModelCheckpoint('tf_inceptionV3-{epoch:03d}.h5', verbose=1, monitor='val_binary_accuracy', save_best_only=True, mode='max')
lr_history = model.fit(train_it, steps_per_epoch=nb_train_samples // bs, validation_data=val_it, validation_steps=nb_val_samples // bs, epochs=nb_epochs, verbose=1, callbacks=[es, checkpoint])
summarize_diagnostics(lr_history, file_name="Learning_Curve_tf")

# 타겟-예측 raw data 저장
test_it.reset()
preds = model.predict(test_it, steps=nb_test_samples, verbose=1)
#pred_labels = list(numpy.argmax(preds, axis=-1))
pred_labels = list(tf.greater(preds, 0.5))
pred_labels = list(chain.from_iterable(pred_labels))
temp = numpy.vstack([test_it.classes, pred_labels])
numpy.savetxt("target_prediction_tfLearning.csv", temp, delimiter=",", fmt='%s')

# ROC 커브
fpr, tpr, _ = roc_curve(test_it.classes, pred_labels)
roc_auc = auc(fpr, tpr)
fig = plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - tfLearning')
plt.legend(loc="lower right")
plt.show()
plt.savefig('ROCCurve_tfLearning.png')

#Confution Matrix and Classification Report
print('Confusion Matrix')
print(confusion_matrix(test_it.classes, pred_labels))
print('Classification Report')
target_names = ['benign', 'cancer']
print(classification_report(test_it.classes, pred_labels, target_names=target_names))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_it.classes, pred_labels)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_it.classes, pred_labels)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_it.classes, pred_labels)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_it.classes, pred_labels)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(test_it.classes, pred_labels)
print('Cohens kappa: %f' % kappa)
# Scores
test_results = numpy.hstack([accuracy, precision, recall, f1, kappa])
numpy.savetxt("test_results_tfLearning.csv", test_results, delimiter=",", fmt='%s')

#Confution Matrix and Classification Report
target_names = ['benign', 'cancer']
cm = confusion_matrix(test_it.classes, pred_labels)
plot_confusion_matrix(cm=cm,
                      normalize=False,
                      target_names=target_names,
                      title="Confusion Matrix",
                      file_name="Confusion_Matrix_tf")

train_it.reset()
val_it.reset()

for layer in base_model.layers[249:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

if opt == 'sgd':
    tune_opti = SGD(lr=tune_lrs, decay=decays, momentum=0.9)
elif opt == 'adam':
    tune_opti = Adam(lr=tune_lrs, decay=decays)

model.compile(optimizer=tune_opti, loss='binary_crossentropy', metrics=['binary_accuracy', FN_score])

# Learning
es = EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=15, min_delta=0.001, restore_best_weights=True)
checkpoint = ModelCheckpoint('FineTuning-{epoch:03d}.h5', verbose=1, monitor='val_binary_accuracy', save_best_only=True, mode='max')
lr_history = model.fit(train_it, steps_per_epoch=nb_train_samples // bs, validation_data=val_it, validation_steps=nb_val_samples // bs, epochs=nb_epochs, verbose=1, callbacks=[es, checkpoint])
summarize_diagnostics(lr_history, file_name="Learning_Curve_finetuning")

# 타겟-예측 raw data 저장
test_it.reset()
preds = model.predict(test_it, steps=nb_test_samples, verbose=1)
pred_labels = list(tf.greater(preds, 0.5))
pred_labels = list(chain.from_iterable(pred_labels))
temp = numpy.vstack([test_it.classes, pred_labels])
numpy.savetxt("target_prediction_FineTuning.csv", temp, delimiter=",", fmt='%s')

# ROC 커브
fpr, tpr, _ = roc_curve(test_it.classes, pred_labels)
roc_auc = auc(fpr, tpr)
fig = plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - FineTuning')
plt.legend(loc="lower right")
plt.show()
plt.savefig('ROCCurve_FineTuning.png')

#Confution Matrix and Classification Report
print('Confusion Matrix')
print(confusion_matrix(test_it.classes, pred_labels))
print('Classification Report')
target_names = ['benign', 'cancer']
print(classification_report(test_it.classes, pred_labels, target_names=target_names))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_it.classes, pred_labels)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_it.classes, pred_labels)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_it.classes, pred_labels)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_it.classes, pred_labels)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(test_it.classes, pred_labels)
print('Cohens kappa: %f' % kappa)
# Scores
test_results = numpy.hstack([accuracy, precision, recall, f1, kappa])
numpy.savetxt("test_results_FineTuning.csv", test_results, delimiter=",", fmt='%s')

#Confution Matrix and Classification Report
target_names = ['benign', 'cancer']
cm = confusion_matrix(test_it.classes, pred_labels)
plot_confusion_matrix(cm=cm,
                      normalize=False,
                      target_names=target_names,
                      title="Confusion Matrix",
                      file_name="Confusion_Matrix_finetuning")
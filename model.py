import numpy as np
from sklearn import svm
import sklearn
import os

from openface.model import create_model
from utils import read_embeddings, get_classes_from_data, write_classifier


def build_model():
  model = create_model()
  model.load_weights('./openface/weights/nn4.small2.v1.h5')
  return model


def build_classifier(data_path):
  print('Constructing classifier...')

  classes = sorted(get_classes_from_data(data_path))

  if len(classes) <= 1:
    print("Not building classifier since we don't have enough faces")
    return

  class_to_num = {x: i for i, x in enumerate(classes)}
  X = None
  Y = None

  for i, aclass in enumerate(classes):
    embeddings = read_embeddings(os.path.join(data_path, aclass, 'embedding.h5'))
    X = embeddings if X is None else np.concatenate((X, embeddings), axis=0)
    labels = np.repeat(class_to_num[aclass], len(embeddings))
    Y = labels if Y is None else np.concatenate((Y, labels), axis=0)

  model = svm.SVC(kernel='linear', probability=True)
  model.fit(X, Y)

  write_classifier(os.path.join(data_path, 'classifier.pickle'), model, classes)

  return (model, classes, X, Y)

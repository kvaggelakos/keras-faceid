from openface.model import create_model

def build_model():
  model = create_model()
  model.load_weights('./openface/weights/nn4.small2.v1.h5')
  return model

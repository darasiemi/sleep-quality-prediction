import pickle

def load_model(model_path):

    with open(model_path, 'rb') as f_in:
        pipeline = pickle.load(f_in)

    return pipeline
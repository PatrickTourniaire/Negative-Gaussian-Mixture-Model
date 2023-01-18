import pickle

def save_object(obj: any, repo: str, file: str):
    # Path constant to save the object
    PATH = f'{repo}/{file}.pkl'

    # Save as a pickle file
    with open(PATH, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(repo: str, file: str):
    # Path constant to save the object
    PATH = f'{repo}/{file}.pkl'

    with open(PATH, 'rb') as f:
        return pickle.load(f)
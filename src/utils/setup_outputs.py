import os

if __name__ == '__main__':
    base_repo = '/exports/eddie/scratch/s1900878/experiment_out'
    base_repo_subs = [
        'data_plots',
        'models',
        'saved_models',
        'sequences'
    ]

    for sub in base_repo_subs:
        path = f'{base_repo}/{sub}/'
        
        if not os.path.isdir(os.path.abspath(path)):
            os.makedirs(os.path.abspath(path))

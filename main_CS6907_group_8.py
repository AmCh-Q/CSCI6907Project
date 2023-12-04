# Configuring task, and dataset
# Modified from https://github.com/ardkastrati/EEGEyeNet
from config import config, build_file_name
config['task'] = 'Position_task'
config['dataset'] = 'dots'
config['preprocessing'] = 'min' # or max
config['feature_extraction'] = True # must be set to True for ML_models operating on feature extracted data
config['include_ML_models'] = True
config['include_DL_models'] = True
config['include_your_models'] = False
config['all_EEG_file'] = build_file_name()

# Add our model and SOTA
import benchmark
from hyperparameters import merge_models
from our_models import Ours_pretrained, EEGViT_pretrained
additional_models = {
    'Position_task': {
        'dots' : {
            'min_temp' : {
                'EEGViT': EEGViT_pretrained,
                'Ours': Ours_pretrained
            }
        }
    }
}
benchmark.all_models = merge_models(benchmark.all_models, additional_models)

# Training and benchmarking the models
# Modified from https://github.com/ardkastrati/EEGEyeNet
import main
main.config = config
main.main()

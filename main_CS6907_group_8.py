# Configuring task, and dataset
from config import config, build_file_name
config['task'] = 'Position_task'
config['dataset'] = 'dots'
config['preprocessing'] = 'min' # or max
config['feature_extraction'] = True # must be set to True for ML_models operating on feature extracted data
config['include_ML_models'] = True
config['include_DL_models'] = True
config['include_your_models'] = True
config['all_EEG_file'] = build_file_name()

# Add our model and SOTA
import benchmark
from hyperparameters import merge_models
from our_models import Ours_Pretrained
additional_models = {
    'Position_task': {
        'dots' : {
            'min_temp' : {
                #'EEGViT': EEGViT_pretrained,
                'Ours': Ours_Pretrained
            }
        }
    }
}
benchmark.all_models = merge_models(main.benchmark.all_models, additional_models)

# Training and benchmarking the models
import main
main.config = config
main.benchmark.all_models = benchmark.all_models
main.main()

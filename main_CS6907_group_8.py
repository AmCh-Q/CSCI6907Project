# Configuring task, and dataset
from config import config, build_file_name
config['task'] = 'Position_task'
config['dataset'] = 'dots'
config['preprocessing'] = 'min'
config['feature_extraction'] = False # our model works on non-feature-extracted data
config['include_ML_models'] = False
config['include_DL_models'] = True
config['include_your_models'] = True
config['all_EEG_file'] = build_file_name()

# Add our model and SOTA
import benchmark
from hyperparameters import merge_models
from EEGViT_pretrained import EEGViT_Pretrained
from our_models import Ours_Pretrained
additional_models = {
    'Position_task': {
        'dots' : {
            'min' : {
                'EEGViT': [EEGViT_Pretrained, {
                    'model_name': "EEGViT",
                    'nb_models': 1,
                    'batch_size': 64,
                    'n_epoch': 15,
                    'learning_rate': 1e-4,
                    'vit_model_name': "google/vit-base-patch16-224"}],
                'Ours_Pretrained': [Ours_Pretrained, {
                    'model_name': "Ours_Pretrained",
                    'nb_models': 1,
                    'batch_size': 64,
                    'n_epoch': 15,
                    'learning_rate': 1e-4,
                    'vit_model_name': "google/vit-base-patch16-224"}]
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

from torch import nn
import transformers
from DL_Models.torch_models.torch_utils.dataloader import create_dataloader
from DL_Models.torch_models.torch_utils.training import test_loop

class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, trainX, trainY, validX, validY):
        #TODO

    def predict(self, testX):
        #TODO

    def save(self, path):
        import pickle
        filename = path + self.model_name + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, path):
        import pickle
        filename = path + self.model_name + '.sav'
        self.model = pickle.load(open(filename, 'rb'))

class Ours_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "Ours_pretrained"
        self.vit_model_name = "google/vit-base-patch16-224"
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1,16),
            stride=(1,16),
            padding=(0,6),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        config = transformers.ViTConfig.from_pretrained(self.vit_model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (129,32)})
        config.update({'patch_size': (129,1)})
        self.model = transformers.ViTForImageClassification.from_pretrained(
            self.vit_model_name, config=config, ignore_mismatched_sizes=True)
        self.model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            256, 768,
            kernel_size=(129,1), stride=(129,1), padding=(0,0),
            groups=256)
        self.model.classifier = nn.Sequential(
            nn.Linear(768,1000,bias=True),
            nn.Dropout(p=0.1),
            nn.Linear(1000,2,bias=True))

    def fit(self, trainX, trainY, validX, validY):
        #TODO
        # Create dataloaders
        trainX = np.transpose(trainX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        validX = np.transpose(validX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        train_dataloader = create_dataloader(trainX, trainY, self.batch_size, self.model_name)
        validation_dataloader = create_dataloader(validX, validY, self.batch_size, self.model_name)
        # Fit the models
        for i in range(self.nb_models):
            logging.info("------------------------------------------------------------------------------------")
            logging.info('Start fitting model number {}/{} ...'.format(i+1, self.nb_models))
            model = self.model(loss = self.loss, model_number=i, batch_size=self.batch_size, **self.model_params)
            model.fit(train_dataloader, validation_dataloader)
            self.models.append(model)
            logging.info('Finished fitting model number {}/{} ...'.format(i+1, self.nb_models))

    def predict(self, testX):
        #TODO

    def save(self, path):
        import pickle
        filename = path + self.model_name + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, path):
        import pickle
        filename = path + self.model_name + '.sav'
        self.model = pickle.load(open(filename, 'rb'))

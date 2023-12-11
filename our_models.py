import torch
from torch import nn
import numpy as np
import transformers
import logging
from tqdm import tqdm
from DL_Models.torch_models.torch_utils.dataloader import create_dataloader

class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, trainX, trainY, validX, validY):
        pass

    def predict(self, testX):
        pass

    def save(self, path):
        import pickle
        filename = path + self.model_name + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, path):
        import pickle
        filename = path + self.model_name + '.sav'
        self.model = pickle.load(open(filename, 'rb'))

class Ours_Pretrained(nn.Module):
    def __init__(self, model_name = "Ours_Pretrained", nb_models = 5, batch_size = 64, n_epoch = 15, learning_rate = 1e-4, vit_model_name = "google/vit-base-patch16-224"):
        super().__init__()
        self.model_name = model_name
        self.nb_models = nb_models
        self.models = []

        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.vit_model_name = vit_model_name

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1,16),
            stride=(1,16),
            padding=(0,6),
            bias=False)
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        self.model_configs = transformers.ViTConfig.from_pretrained(self.vit_model_name)
        self.model_configs.update({'num_channels': 256})
        self.model_configs.update({'image_size': (129,32)})
        self.model_configs.update({'patch_size': (129,1)})
        if torch.cuda.is_available():
            gpu_id = 0    # Change this to the desired GPU ID if you have multiple GPUs
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits
        return x

    def create_model(self):
        model = transformers.ViTForImageClassification.from_pretrained(
            self.vit_model_name, config=self.model_configs, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            256, 768,
            kernel_size=(129,1), stride=(129,1), padding=(0,0),
            groups=256)
        model.classifier = nn.Sequential(
            nn.Linear(768,1000,bias=True),
            nn.Dropout(p=0.1),
            nn.Linear(1000,2,bias=True))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.1)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        return model

    def fit(self, trainX, trainY, validX, validY):
        # Create dataloaders
        trainX = np.transpose(trainX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        validX = np.transpose(validX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        train_dataloader = create_dataloader(trainX, trainY, self.batch_size, self.model_name)
        validation_dataloader = create_dataloader(validX, validY, self.batch_size, self.model_name)
        # Fit the models
        for i in range(self.nb_models):
            logging.info("------------------------------------------------------------------------------------")
            logging.info('Start fitting model number {}/{} ...'.format(i+1, self.nb_models))
            model = self.create_model()
            for epoch in range(self.n_epoch):
                logging.info("-------------------------------")
                logging.info(f"Epoch {epoch+1}")
                # Train the model
                model.train()
                epoch_train_loss = 0.0
                for index, inputs, targets in tqdm(enumerate(train_dataloader)):
                    # Move the inputs and targets to the GPU (if available)
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    # Compute the outputs and loss for the current batch
                    self.optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets.squeeze())
                    # Compute the gradients and update the parameters
                    loss.backward()
                    self.optimizer.step()
                    epoch_train_loss += loss.item()
                    # Print the loss and accuracy for the current batch
                    if index in printBatchNumList:
                        print(f"Epoch {epoch+1}, Batch {index}, Training Loss: {loss.item()}", end='')
                epoch_train_loss /= len(train_dataloader)
                logging.info(f"Avg training loss: {epoch_train_loss:>7f}")
                print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}", end='')
                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for index, inputs, targets in enumerate(validation_dataloader):
                        # Move the inputs and targets to the GPU (if available)
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        # Compute the outputs and loss for the current batch
                        outputs = model(inputs)
                        loss = criterion(outputs.squeeze(), targets.squeeze())
                        val_loss += loss.item()
                    val_loss /= len(validation_dataloader)
                logging.info(f"Avg validation loss: {val_loss:>8f}")
                print(f"Epoch {epoch+1}, Val Loss: {val_loss}")
            self.models.append(model)
            logging.info('Finished fitting model number {}/{} ...'.format(i+1, self.nb_models))

    def predict(self, testX):
        testX = np.transpose(testX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        a,b,c = testX.shape
        a = self.batch_size - a % self.batch_size
        dummy = np.zeros((a,b,c))
        testX = np.concatenate((testX, dummy)) # TO ADD batch_size - testX.shape[0]%batch_size
        test_dataloader = create_dataloader(testX, testX, self.batch_size, self.model_name, drop_last=False)
        pred = None
        for model in self.models:
            for index, inputs, _ in enumerate(test_dataloader):
                # Move the inputs to the GPU (if available)
                inputs = inputs.to(self.device)
                # Compute the outputs
                outputs = model(inputs)
                if batch == 0:
                    all_pred = outputs.cpu()
                else:
                    all_pred = torch.cat((all_pred, outputs.cpu()))
            if pred is not None:
                pred += all_pred.data.numpy()
            else:
                pred = all_pred.data.numpy()
        pred = pred[:-a]
        return pred / len(self.models)

    def save(self, path):
        for i, model in enumerate(self.models):
            ckpt_dir = path + self.model_name + '_nb_{}_'.format(i) + '.pth'
            torch.save(model.state_dict(), ckpt_dir)

    def load(self, path):
        self.models = []
        for file in os.listdir(path):
            if not self.load_file_pattern.match(file):
                continue
            # These 2 lines are needed for torch to load
            logging.info(f"Loading model nb from file {file} and predict with it")
            model = self.create_model()  # model = TheModelClass(*args, **kwargs)
            #print(path + file)
            model.load_state_dict(torch.load(path + file))  # model.load_state_dict(torch.load(PATH))
            model.eval()  # needed before prediction
            self.models.append(model)

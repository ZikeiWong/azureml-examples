import os
import time
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl

import torch_nebula as tn


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":

    os.environ["DLTS_NUM_WORKER"] = "1"
    os.environ["FC_TASK_INDEX"] = "0"
    os.environ["FC_TASKROLE_NAME"] = "worker"
    os.environ["DLTS_JOB_ID"] = "test" 

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])
    train = Subset(train, range(1000))
    val = Subset(val, range(100))

    autoencoder = LitAutoEncoder()

    persisted_path = "/tmp/tier3/default_path"
    tn.init(persistent_storage_path=persisted_path)
    #config_params = dict()
    #storage_options = dict()
    #config_params["persistent_storage_path"] = persisted_path
    #config_params["persistent_time_interval"] = 5

    #nebula_callback = tn.NebulaCallback(every_n_epochs=5, config_params=config_params)
    early_stop_callback = EarlyStopping(monitor="train_loss", mode="min", min_delta=0.001)

    # 1. Save during training.
    print("Conditional save test...")
    trainer = pl.Trainer(
        gpus=0, 
        strategy="ddp", 
        max_epochs=20, 
        plugins=[tn.NebulaCheckpointIO()]
        #, callbacks=[nebula_callback]
    )
    trainer.fit(autoencoder, DataLoader(train), DataLoader(val))

    # 2. Manual save.
    print("Manual save test...")
    storage_options = {}
    storage_options["is_best"] = True
    storage_options["persist_path"] = "/tmp/tier3/default_path"

    trainer.save_checkpoint(filepath="epoch=20-step=20000.ckpt", storage_options=storage_options)

    saved_ckpt = "epoch=20-step=20000.ckpt"

    print("load before training")
    # 3. Load before training /nebula load
    trainer = pl.Trainer(
        gpus=0,
        strategy="ddp",
        max_epochs=50,
        plugins=[tn.NebulaCheckpointIO()],
        callbacks=[nebula_callback, early_stop_callback],
    )
    trainer.fit(autoencoder, DataLoader(train), DataLoader(val), ckpt_path=saved_ckpt)

    # 4. Load from path.
    ckpts = tn.list_checkpoints()
    js = json.dumps(ckpts, sort_keys=True, indent=4, separators=(",", ":"))
    print("\n*********** list checkpoints results**********\n")
    print(js)

    latest_ckpt_path = tn.get_latest_checkpoint_path("checkpoint", persisted_path)
    print("\n*********** get latest checkpoints path **********\n")
    print(latest_ckpt_path)
    LitAutoEncoder.load_from_checkpoint(checkpoint_path=latest_ckpt_path)


    
    print("\n*********** load checkpoints successfully **********\n")

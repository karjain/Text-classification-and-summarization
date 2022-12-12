import os
import torch
import gdown


class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        code_dir = os.getcwd()
        model_dir = os.path.join(os.path.split(code_dir)[0], 'Model')
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def __call__(self, current_valid_loss, epoch, model, model_save_name):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            torch.save(
                model.state_dict(),
                os.path.join(self.model_dir, model_save_name)
            )


def avail_data(data_dir):
    cur_dir = os.getcwd()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)
    file_id_mapping = {
        "Combined_Headlines.json": "1NSpIABIfwZ3E1As6XVbNCX3do0D9MucI",
        "sarcastic_output.json": "19dS1iQ51oxRmiEkoArWUwW6BqXYDJGuo",
        "glove.6B.50d.txt": "1IJE5wg2kvsUVKagg6U2hoCyrTxZlaAUe"
    }
    for key, value in file_id_mapping.items():
        if not os.path.exists(os.path.join(data_dir, key)):
            key_output = gdown.download(id=value, quiet=False)
            if key_output != key:
                print(f"{key} could not be downloaded")
    os.chdir(cur_dir)


def avail_models(model_dir):
    cur_dir = os.getcwd()
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    os.chdir(model_dir)
    file_id_mapping = {
        "trans-roberta-model.pt": "1B4LyKm9qiTXXHPRPhOF8IlcXki0L65ae",
        "t5-small-headline-generator.zip": "1HB6hN2PYnAFNplfhVB2JNvr1Oym_LsWR"
    }
    for key, value in file_id_mapping.items():
        if not os.path.exists(os.path.join(model_dir, key)):
            key_output = gdown.download(id=value, quiet=False)
            if key_output != key:
                print(f"{key} could not be downloaded")
                break
        if key[-4:] == ".zip":
            savedmodel_dir = os.path.join(model_dir, "savedmodel")
            if not os.path.exists(savedmodel_dir):
                os.mkdir(savedmodel_dir)
            if not os.path.exists(os.path.join(savedmodel_dir, key[:-4])):
                os.system(f"unzip {key} -d {savedmodel_dir}")
            os.unlink(key)
    os.chdir(cur_dir)

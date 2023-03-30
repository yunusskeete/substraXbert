import os
import pathlib
import numpy as np
import substratools as tools


class TrecOpener(tools.Opener):
    def fake_data(self, n_samples=None):
        N_SAMPLES = n_samples if n_samples and n_samples <= 100 else 100

        fake_input_ids = np.random.randint(29607+1, size=(N_SAMPLES, 512))

        fake_attention_mask = np.random.randint(0, 1+1, size=(N_SAMPLES, 512))

        fake_labels = np.random.randint(6, size=N_SAMPLES) + 1

        data = {
            "input_ids": fake_input_ids,
            "attention_mask": fake_attention_mask,
            "labels": fake_labels,
        }

        return data

    def get_data(self, folders):
        # get npy files
        p = pathlib.Path(folders[0])
        ids_data_path = os.path.join(p, list(p.glob("*_ids.npy"))[0])
        attention_mask_data_path = os.path.join(p, list(p.glob("*_attention_mask.npy"))[0])
        labels_data_path = os.path.join(p, list(p.glob("*_labels.npy"))[0])

        # load data
        data = {
            "input_ids": np.load(ids_data_path),
            "attention_mask": np.load(attention_mask_data_path),
            "labels": np.load(labels_data_path),
        }

        return data
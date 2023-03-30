"""
===================================
Using Torch FedAvg on TREC dataset with BERT
===================================

This example illustrates the basic usage of SubstraFL and proposes Federated Learning using the Federated Averaging strategy on the `TREC Dataset for natural language classification <hhttps://huggingface.co/datasets/trec>`__ using PyTorch.
Text REtrieval Conference (TREC) Question Classification dataset contains 5500 labeled questions in training set and another 500 for test set.
In this example, we work on text strings with an average length of each sentence of 10 and avocabulary size of 8700.
This is a classification problem aiming to classify the questions from a set of 6 class labels.

SubstraFL can be used with any machine learning framework (PyTorch, Tensorflow, Scikit-Learn, etc). 

However a specific interface has been developed for PyTorch which makes writing PyTorch code simpler than with other frameworks. This example here uses the specific PyTorch interface.

This example does not use a deployed platform of Substra and runs in local mode.

To run this example, you have two options:

- **Recommended option**: use a hosted Jupyter notebook. With this option you don't have to install anything, just run the notebook.
  To access the hosted notebook, scroll down at the bottom of this page and click on the **Launch Binder** button.
- **Run the example locally**. To do that you need to download and unzip the assets needed to run it in the same
  directory as used this example.

   .. only:: builder_html or readthedocs

      :download:`assets required to run this example <../../../../../tmp/torch_fedavg_assets.zip>`

  * Please ensure to have all the libraries installed. A *requirements.txt* file is included in the zip file, where you can run the command ``pip install -r requirements.txt`` to install them.
  * **Substra** and **SubstraFL** should already be installed. If not follow the instructions described here: :ref:`substrafl_doc/substrafl_overview:Installation`.


"""
import shutil
import os

folder_path = "./local-worker"
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
folder_path = "./tmp"
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)


# %%
# Setup
# *****
#
# This examples runs with three organizations. Two organizations provide datasets, while a third
# one provides the algorithm.
#
# In the following code cell, we define the different organizations needed for our FL experiment.


from substra import Client

N_CLIENTS = 3

# Every computation will run in `subprocess` mode, where everything runs locally in Python
# subprocesses.
# Ohers backend_types are:
# "docker" mode where computations run locally in docker containers
# "remote" mode where computations run remotely (you need to have a deployed platform for that)
clients = [Client(backend_type="subprocess") for _ in range(N_CLIENTS)]
clients = {client.organization_info().organization_id: client for client in clients}
# To run in remote mode you have to also use the function `Client.login(username, password)`

# Store organization IDs
ORGS_ID = list(clients.keys())
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data providers orgs are the two last organizations.

# %%
# Data and metrics
# ****************

# %%
# Data preparation
# ================
#
# This section downloads (if needed) the **MNIST dataset** using the `torchvision library
# <https://pytorch.org/vision/stable/index.html>`__.
# It extracts the images from the raw files and locally creates a folder for each
# organization.
#
# Each organization will have access to half the training data and half the test data (which
# corresponds to **30,000**
# images for training and **5,000** for testing each).

import pathlib
from torch_fedavg_assets.dataset.trec_dataset import setup_trec

# sphinx_gallery_thumbnail_path = 'static/example_thumbnail/mnist.png'

# Create the temporary directory for generated data
(pathlib.Path.cwd() / "tmp").mkdir(exist_ok=True)
data_path = pathlib.Path.cwd() / "tmp" / "data_trec"

setup_trec(data_path, len(DATA_PROVIDER_ORGS_ID))

# %%
# Dataset registration
# ====================
#
# A :ref:`documentation/concepts:Dataset` is composed of an **opener**, which is a Python script that can load
# the data from the files in memory and a description markdown file.
# The :ref:`documentation/concepts:Dataset` object itself does not contain the data. The proper asset that contains the
# data is the **datasample asset**.
#
# A **datasample** contains a local path to the data. A datasample can be linked to a dataset in order to add data to a
# dataset.
#
# Data privacy is a key concept for Federated Learning experiments. That is why we set
# :ref:`documentation/concepts:Permissions` for :ref:`documentation/concepts:Assets` to determine how each organization can access a specific asset.
#
# Note that metadata such as the assets' creation date and the asset owner are visible to all the organizations of a
# network.

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

assets_directory = pathlib.Path.cwd() / "torch_fedavg_assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):

    client = clients[org_id]

    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    dataset = DatasetSpec(
        name="TREC",
        type="npy",
        data_opener=assets_directory / "dataset" / "trec_opener.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing dataset key"

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        test_only=False,
        path=data_path / f"org_{i+1}" / "train",
    )
    train_datasample_keys[org_id] = client.add_data_sample(data_sample)

    # Add the testing data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        test_only=True,
        path=data_path / f"org_{i+1}" / "test",
    )
    test_datasample_keys[org_id] = client.add_data_sample(data_sample)


from datasets import load_dataset # pip install datasets

# load the TREC dataset
# trec = load_dataset('trec', split='train')
trec = load_dataset('trec', split='train[:1000]')

# %%
# Metric registration
# ===================
#
# A metric is a function used to evaluate the performance of your model on one or several
# **datasamples**.
#
# To add a metric, you need to define a function that computes and returns a performance
# from the datasamples (as returned by the opener) and the predictions_path (to be loaded within the function).
#
# When using a Torch SubstraFL algorithm, the predictions are saved in the `predict` function in numpy format
# so that you can simply load them using `np.load`.
#
# After defining the metrics, dependencies, and permissions, we use the `add_metric` function to register the metric.
# This metric will be used on the test datasamples to evaluate the model performances.

import torch
import numpy as np

from substrafl.dependency import Dependency
from substrafl.remote.register import add_metric

permissions_metric = Permissions(
    public=False, authorized_ids=[ALGO_ORG_ID] + DATA_PROVIDER_ORGS_ID
)

# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
metric_deps = Dependency(pypi_dependencies=["numpy==1.23.5"])


def accuracy(datasamples, predictions_path):

    outputs = np.load(predictions_path)
    labels = datasamples["labels"]

    _, preds = np.argmax(outputs, dim=1)

    # I don't think we need to one-hot encode labels AT ALL

    _, targets = np.argmax(labels, dim=1)
    correct = preds == targets
    acc = sum(correct) / len(correct)
    
    return acc



metric_key = add_metric(
    client=clients[ALGO_ORG_ID],
    metric_function=accuracy,
    permissions=permissions_metric,
    dependencies=metric_deps,
)


# %%
# Specify the machine learning components
# ***************************************
# This section uses the PyTorch based SubstraFL API to simplify the definition of machine learning components.
# However, SubstraFL is compatible with any machine learning framework.
#
#
# In this section, you will:
#
# - Register a model and its dependencies
# - Specify the federated learning strategy
# - Specify the training and aggregation nodes
# - Specify the test nodes
# - Actually run the computations


# %%
# Model definition
# ================
#
# We choose to use a classic torch CNN as the model to train. The model structure is defined by the user independently
# of SubstraFL.

# from transformers import AdamW
from torch.optim import AdamW

seed = 42
torch.manual_seed(seed)


from transformers import BertForSequenceClassification, BertConfig

config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = max(trec['coarse_label'])+1
model = BertForSequenceClassification(config)#.to(device)

# # activate training mode of model
# model.train()

# initialize adam optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# %%
# Specifying on how much data to train
# ====================================
#
# To specify on how much data to train at each round, we use the `index_generator` object.
# We specify the batch size and the number of batches to consider for each round (called `num_updates`).
# See :ref:`substrafl_doc/substrafl_overview:Index Generator` for more details.


from substrafl.index_generator import NpIndexGenerator

# Number of model updates between each FL strategy aggregation.
"""                 WHAT DOES THIS DO?????                  """
NUM_UPDATES = 2 # 100

# Number of samples per update.
BATCH_SIZE = 32 # 64

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)

# %%
# Torch Dataset definition
# ==========================
#
# This torch Dataset is used to preprocess the data using the `__getitem__` function.
#
# This torch Dataset needs to have a specific `__init__` signature, that must contain (self, datasamples, is_inference).
#
# The `__getitem__` function is expected to return (inputs, outputs) if `is_inference` is `False`, else only the inputs.
# This behavior can be changed by re-writing the `_local_train` or `predict` methods.


class TrecDataset(torch.utils.data.Dataset):
    def __init__(self, datasamples, is_inference):
        self.input_ids = datasamples["input_ids"]
        self.attention_mask = datasamples["attention_mask"]
        self.labels = datasamples["labels"]

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

    def __len__(self):
        return len(self.labels)


# %%
# SubstraFL algo definition
# ==========================
#
# A SubstraFL Algo gathers all the defined elements that run locally in each organization.
# This is the only SubstraFL object that is framework specific (here PyTorch specific).
#
# The `TorchDataset` is passed **as a class** to the `Torch algorithm <substrafl_doc/api/algorithms:Torch Algorithms>`_.
# Indeed, this `TorchDataset` will be instantiated directly on the data provider organization.


from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.exceptions import BatchSizeNotFoundError
from substrafl.exceptions import OptimizerValueError


class MyTorchAlgo(TorchAlgo):

    def _local_predict(self, predict_dataset: torch.utils.data.Dataset, predictions_path):
        """Executes the following operations:

            * Create the torch dataloader using the index generator batch size.
            * Sets the model to `eval` mode
            * Save the predictions using the
              :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo._save_predictions` function.

        Args:
            predict_dataset (torch.utils.data.Dataset): predict_dataset build from the x returned by the opener.

        Important:
            The onus is on the user to ``save`` the compute predictions. Substrafl provides the
            :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo._save_predictions` to do so.
            The user can load those predictions from a metric file with the command:
            ``y_pred = np.load(inputs['predictions'])``.

        Raises:
            BatchSizeNotFoundError: No default batch size have been found to perform local prediction.
                Please overwrite the predict function of your algorithm.
        """
        if self._index_generator is not None:
            predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=self._index_generator.batch_size)
        else:
            raise BatchSizeNotFoundError(
                "No default batch size has been found to perform local prediction. "
                "Please overwrite the _local_predict function of your algorithm."
            )

        self._model.eval()

        predictions = torch.Tensor([])
        with torch.inference_mode():
            for batch in predict_loader:
                batch_mps = {
                    'input_ids': batch['input_ids'].to(self._device),
                    'attention_mask': batch['attention_mask'].to(self._device),
                    'labels': batch['labels'].to(self._device)
                }
                outputs = self._model(**batch_mps)
                preds = outputs.logits
                predictions = torch.cat((predictions, preds), 0)

        predictions = predictions.cpu().detach()
        self._save_predictions(predictions, predictions_path)

    def _local_train(
            self,
            train_dataset: torch.utils.data.Dataset,
    ):
        """Local train method. Contains the local training loop.

        Train the model on ``num_updates`` minibatches, using the ``self._index_generator generator`` as batch sampler
        for the torch dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): train_dataset build from the x and y returned by the opener.

        Important:

            You must use ``next(self._index_generator)`` as batch sampler,
            to ensure that the batches you are using are correct between 2 rounds
            of the federated learning strategy.

        Example:

            .. code-block:: python

                # Create torch dataloader
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)

                for x_batch, y_batch in train_data_loader:

                    # Forward pass
                    y_pred = self._model(x_batch)

                    # Compute Loss
                    loss = self._criterion(y_pred, y_batch)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    if self._scheduler is not None:
                        self._scheduler.step()
        """
        if self._optimizer is None:
            raise OptimizerValueError(
                "No optimizer found. Either give one or overwrite the _local_train method from the used torch"
                "algorithm."
            )

        # Create torch dataloader
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)


        for batch in train_data_loader:
            batch_mps = {
                'input_ids': batch['input_ids'].to(self._device),
                'attention_mask': batch['attention_mask'].to(self._device),
                'labels': batch['labels'].to(self._device)
            }

            # (Forward pass)
            # train model on batch and return outputs (incl. loss)
            outputs = self._model(**batch_mps)

            # (Compute Loss)
            # extract loss
            # loss = outputs.loss
            loss = self._criterion(outputs.logits, batch_mps["labels"])


            # initialize calculated gradients (from prev step)
            self._optimizer.zero_grad()
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            self._optimizer.step()

            if self._scheduler is not None:
                self._scheduler.step()      

class MyTorchFedAvgAlgo(MyTorchAlgo, TorchFedAvgAlgo):
    pass

class MyAlgo(MyTorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TrecDataset,
            seed=seed,
        )


# %%
# Federated Learning strategies
# =============================
#
# A FL strategy specifies how to train a model on distributed data.
# The most well known strategy is the Federated Averaging strategy: train locally a model on every organization,
# then aggregate the weight updates from every organization, and then apply locally at each organization the averaged
# updates.


from substrafl.strategies import FedAvg

strategy = FedAvg()

# %%
# Where to train where to aggregate
# =================================
#
# We specify on which data we want to train our model, using the :ref:`substrafl_doc/api/nodes:TrainDataNode` object.
# Here we train on the two datasets that we have registered earlier.
#
# The :ref:`substrafl_doc/api/nodes:AggregationNode` specifies the organization on which the aggregation operation
# will be computed.

from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


aggregation_node = AggregationNode(ALGO_ORG_ID)

train_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:

    # Create the Train Data Node (or training task) and save it in a list
    train_data_node = TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    train_data_nodes.append(train_data_node)

# %%
# Where and when to test
# ======================
#
# With the same logic as the train nodes, we create :ref:`substrafl_doc/api/nodes:TestDataNode` to specify on which
# data we want to test our model.
#
# The :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy` defines where and at which frequency we
# evaluate the model, using the given metric(s) that you registered in a previous section.


from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy


test_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:

    # Create the Test Data Node (or testing task) and save it in a list
    test_data_node = TestDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        test_data_sample_keys=[test_datasample_keys[org_id]],
        metric_keys=[metric_key],
    )
    test_data_nodes.append(test_data_node)

# Test at the end of every round
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=1)

# %%
# Running the experiment
# **********************
#
# We now have all the necessary objects to launch our experiment. Please see a summary below of all the objects we created so far:
#
# - A :ref:`documentation/references/sdk:Client` to add or retrieve the assets of our experiment, using their keys to
#   identify them.
# - An `Torch algorithm <substrafl_doc/api/algorithms:Torch Algorithms>`_ to define the training parameters *(optimizer, train
#   function, predict function, etc...)*.
# - A `Federated Strategy <substrafl_doc/api/strategies:Strategies>`_, to specify how to train the model on
#   distributed data.
# - `Train data nodes <substrafl_doc/api/nodes:TrainDataNode>`_ to indicate on which data to train.
# - An :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy`, to define where and at which frequency we
#   evaluate the model.
# - An :ref:`substrafl_doc/api/nodes:AggregationNode`, to specify the organization on which the aggregation operation
#   will be computed.
# - The **number of rounds**, a round being defined by a local training step followed by an aggregation operation.
# - An **experiment folder** to save a summary of the operation made.
# - The :ref:`substrafl_doc/api/dependency:Dependency` to define the libraries on which the experiment needs to run.

from substrafl.experiment import execute_experiment

# A round is defined by a local training step followed by an aggregation operation
NUM_ROUNDS = 3

# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
algo_deps = Dependency(pypi_dependencies=["numpy==1.23.5", "torch==2.0.0"])

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    algo=MyAlgo(),
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=algo_deps,
)

# %%
# Explore the results
# *******************

# %%
# List results
# ============


import pandas as pd

performances_df = pd.DataFrame(client.get_performances(compute_plan.key).dict())
print("\nPerformance Table: \n")
print(performances_df[["worker", "round_idx", "performance"]])

# %%
# Plot results
# ============

import matplotlib.pyplot as plt

plt.title("Test dataset results")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")

for id in DATA_PROVIDER_ORGS_ID:
    df = performances_df.query(f"worker == '{id}'")
    plt.plot(df["round_idx"], df["performance"], label=id)

plt.legend(loc="lower right")
plt.show()

# %%
# Download a model
# ================
#
# After the experiment, you might be interested in downloading your trained model.
# To do so, you will need the source code in order to reload your code architecture in memory.
# You have the option to choose the client and the round you are interested in downloading.
#
# If `round_idx` is set to `None`, the last round will be selected by default.

from substrafl.model_loading import download_algo_files
from substrafl.model_loading import load_algo

client_to_dowload_from = DATA_PROVIDER_ORGS_ID[0]
round_idx = None

algo_files_folder = str(pathlib.Path.cwd() / "tmp" / "algo_files")

download_algo_files(
    client=clients[client_to_dowload_from],
    compute_plan_key=compute_plan.key,
    round_idx=round_idx,
    dest_folder=algo_files_folder,
)

model = load_algo(input_folder=algo_files_folder).model

print(model)

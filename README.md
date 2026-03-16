# Wine Prediction Model with DVC and GCP

This project demonstrates the use of DVC (Data Version Control) to manage datasets for a simple wine quality prediction model. It is configured to use Google Cloud Storage (GCS) as a remote backend to store and version control large data files that are not suitable for Git.

## Overview

In MLOps, it's crucial to version not just your code but also your data and models. This ensures reproducibility and traceability. This repository uses:
- **Git:** To version control the source code (e.g., `train.py`).
- **DVC:** To version control the dataset (`wine_sample.csv`). The actual data is stored in a remote GCS bucket.
- **GCS (Google Cloud Storage):** As the remote storage for DVC.

The `.dvc` files in this repository (like `data/wine_sample.csv.dvc`) are small pointers that tell DVC where to find the actual data in GCS. These pointers are committed to Git, allowing you to track data changes alongside code changes.

## What is DVC (Data Version Control)?

DVC is an open-source tool designed to bring version control to data and machine learning models. It works on top of Git, allowing you to manage large datasets, models, and experiments with the same best practices you use for code.

### Why Use DVC in MLOps?

- **Handles Large Datasets**: Git is not designed to handle large files. DVC stores data and models in a separate remote storage (like GCS, S3, or a shared server) while Git only tracks small metadata files. This keeps your Git repository small and fast.
- **Ensures Reproducibility**: DVC tracks the exact version of the data and code used in each experiment. This makes your machine learning projects fully reproducible, allowing you to go back to any point in time and recreate your results.
- **Facilitates Collaboration**: DVC makes it easy for teams to share and collaborate on datasets and models. Team members can push and pull data just like they do with code, ensuring everyone is working with the same versions.
- **Experiment Tracking**: DVC helps you track experiments by linking code, data, and hyperparameters to your results. This organized approach is crucial for comparing different experiment runs and understanding what works best.

## Prerequisites

Before you begin, ensure you have the following installed:
- Git
- Python 3.8+
- [DVC](https://dvc.org/doc/install)
- [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install)

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine.

```bash
git clone <your-repository-url>
cd dvc-demo-wine-prediction-model
```

### 2. Create and Activate a Python Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required Python libraries from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Set Up Google Cloud Storage (GCS)

You need a GCS bucket to store your data. If you don't have one, follow these steps.

1.  **Authenticate with GCP:**
    Log in to your Google Cloud account.

    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```

2.  **Create a GCS Bucket:**
    Choose a unique name for your bucket and create it.

    ```bash
    gsutil mb gs://<your-unique-bucket-name>
    ```

### 5. Configure DVC Remote Storage
Tell DVC to use your newly created GCS bucket as the remote storage location. This command configures the `.dvc/config` file, which is tracked by Git.

```bash
# Replace <your-unique-bucket-name> with the name of your GCS bucket
dvc remote add -d myremote gcs://<your-unique-bucket-name>/dvc-store
```
This sets up a remote named `myremote`. The `-d` flag sets it as the default remote.

### 6. Pull the Data
Now that DVC is configured, you can pull the data from the remote storage. This will download the `wine_sample.csv` file into the `data/` directory.

```bash
dvc pull
```

Your project is now fully set up! The `data/` directory contains the dataset, and you are ready to train the model.

## Training the Model

To run the training script, execute the following command:

```bash
python train.py
```

## DVC Workflow for Collaboration

Here’s how you and your colleagues can collaborate using this setup:

- **Getting updates:** When a team member pushes a new version of the data, you can get it by running `git pull` (to get the latest `.dvc` pointer file) followed by `dvc pull` (to download the actual data from GCS).

- **Updating the data:** If you modify the dataset (e.g., add more samples), you need to track the changes with DVC and push them to the remote storage.

  ```bash
  # Tell DVC to track the new version of the data
  dvc add data/wine_sample.csv

  # Push the data to the GCS bucket
  dvc push

  # Commit the updated .dvc file to Git
  git add data/wine_sample.csv.dvc
  git commit -m "Updated wine dataset"
  git push
  ```

Your colleagues can now pull your changes and get the new dataset.

# Evolving Prototype-based Multiple Instance Learning (EP-MIL)

## Overview

**EP-MIL** is a lightweight, computationally efficient framework for Whole Slide Image (WSI) classification. Instead of relying on complex deep learning architectures, this method uses an **Evolutionary Algorithm** to find an optimal set of "prototype" feature vectors for each class.

Classification is performed with a simple nearest-neighbor-like approach: a slide is assigned the label of the closest prototype vector, measured by a standard distance metric like Cosine or Euclidean distance. This makes the inference process extremely fast and interpretable.

The core idea is to treat the set of all prototype vectors as an "individual" in a population that is evolved over generations to maximize classification accuracy on a validation set.

<img width="1280" height="720" alt="overview" src="https://github.com/user-attachments/assets/36e89eaf-6a1b-4945-aae9-baa3b312fcec" />


## Features

* **Fast & Efficient**: Avoids the overhead of training large neural networks. Ideal for rapid prototyping and situations with limited computational resources.

* **Interpretable**: The final model consists of prototype vectors that can be analyzed to understand the feature space of each class.

* **Simple**: The algorithm is based on fundamental concepts of evolutionary computation and distance metrics.

* **Flexible**: Easily adaptable with different distance metrics, evolutionary operators, and hyperparameters.

## How It Works

1. **Feature Extraction**: The pipeline assumes you have already extracted patch-level features from your WSIs using a pre-trained model (e.g., a ResNet, Vision Transformer). These features for each slide should be stored in individual `.npz` files.

2. **Slide Representation**: Each WSI is represented by the **average** of all its patch-level feature vectors.

3. **Evolutionary Optimization**:

   * An initial population of "classifiers" is created. Each classifier is a collection of prototype vectors, randomly sampled from the training data's slide representations.

   * For a set number of generations, the population undergoes evolution:

     * **Evaluation**: Each classifier's fitness is measured by its classification accuracy on the validation set.

     * **Selection**: Better-performing classifiers are selected as "parents" for the next generation.

     * **Crossover & Mutation**: New "child" classifiers are created by combining and slightly modifying the prototype vectors of the parents.

4. **Final Model**: The best-performing set of prototype vectors found throughout the evolution is saved as the final model.

## Repository Setup Guide

Follow these steps to set up the repository and run the code.

### Step 1: Clone the Repository (Or install via pypi: pip install epmil)

First, clone this repository to your local machine.

```
git clone https://github.com/AICAN-Research/EP-MIL
cd EP-MIL
```

### Step 2: Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required Python packages using the provided `requirements.txt` file.

```
pip install -r requirements.txt
```

*(You will need to create a `requirements.txt` file. See section below.)*

### Step 4: Prepare Your Data

Your data needs to be organized as follows:

1. **Feature Files**: A directory containing all your `.npz` feature files. Each `.npz` file should correspond to one WSI and contain a single key named `features` that holds a NumPy array of shape `(num_patches, feature_dim)`.

   ```
   data/
   └── features/
       ├── slide-001.npz
       ├── slide-002.npz
       └── ...
   ```

2. **CSV Files**: You need CSV files that list the slides and their corresponding labels. The script can auto-detect common column names like `File_name`, `slide_id`, `Class`, and `label`. Create one csv for both training and validation sets, and one separate csv for your test set so you can pass them in the command (see usage).

   **`train_slides.csv`**
   ```
   File_name,Class
   slides/slide-001.svs,0
   slides/slide-003.svs,1
   ...
   ```

3. **Validation Set (Optional)**: You can specify a validation set by providing a simple text file that lists the slide IDs (filenames without extensions), one per line.

   **`val_ids.txt`**
   ```
   slide-002
   slide-005
   ...
   ```

## Usage

The script has two main modes: `train` and `predict`.

### Training a Model

To train a new EP-MIL model, use the `train` command.

```
python ep_mil_classifier.py train \
  --features_dir /path/to/your/features \
  --train_csv /path/to/your/train_slides.csv \
  --val_ids_txt /path/to/your/val_ids.txt \
  --save_model_path models/my_first_model.json \
  --generations 50 \
  --population_size 100 \
  --prototypes_per_class 10 \
  --metric euclidean
```

This will run the evolutionary algorithm and save the best model (`my_first_model.json`) and its corresponding label map (`my_first_model_label_map.json`) in the `models/` directory.

### Running Predictions

Once you have a trained model, you can use it to make predictions on a new set of slides with the `predict` command.

```
python ep_mil_classifier.py predict \
  --features_dir /path/to/your/features \
  --test_csv /path/to/your/test_slides.csv \
  --model_checkpoint models/my_first_model.json \
  --output_csv results/predictions.csv
```

This will load the specified model, run inference on the test data, and save the predictions to `results/predictions.csv`.

### Command-Line Arguments

* `--features_dir`: **(Required)** Path to the directory with `.npz` feature files.

* `--train_csv`: (Train) Path to the training data CSV.

* `--test_csv`: (Predict) Path to the test data CSV.

* `--val_ids_txt`: (Train) Optional `.txt` file listing validation slide IDs.

* `--save_model_path`: (Train) Where to save the output model file (`.json`).

* `--model_checkpoint`: (Predict) Path to a trained model file.

* `--output_csv`: (Predict) Where to save the prediction results.

* `--metric`: (Train) Distance metric to use (`cosine` or `euclidean`).

* **Evolutionary Hyperparameters**:

  * `--generations`: Number of generations.

  * `--population_size`: Number of classifiers in the population.

  * `--prototypes_per_class`: Number of prototype vectors for each class.

  * `--mutation_rate`, `--mutation_strength`, `--crossover_rate`.

## Creating `requirements.txt`

The project depends on a few common libraries. Create a file named `requirements.txt` with the following content:

```
numpy
pandas
scikit-learn
tqdm
```

This file is used in **Step 3** to install the necessary packages.

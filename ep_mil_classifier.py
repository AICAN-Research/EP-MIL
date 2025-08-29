"""
=============================================================================
 Evolving Prototype-based Multiple Instance Learning (EP-MIL) for WSI Classification
=============================================================================
By Soroush Oskouei - ISB @ NTNU - AICAN

This script implements a Whole Slide Image classification method using an
evolutionary algorithm to optimize a set of prototype vectors. This approach,
termed EP-MIL, offers a computationally efficient, geometric alternative to
deep learning models for WSI classification.

-----------------------------------------------------------------------------
Algorithm Overview:
-----------------------------------------------------------------------------
1.  **Representation**: A classifier (an "individual" in the evolutionary
    population) is defined by a set of prototype feature vectors for each class.

2.  **Initialization**: The initial population of classifiers is created by
    sampling feature vectors directly from the training data. Each prototype
    is the average feature representation of a slide.

3.  **Fitness Evaluation**: The fitness of a classifier is determined by its
    classification accuracy on a validation set. For each slide, its feature
    vector is compared against all prototypes. The class of the nearest
    prototype (based on a chosen distance metric) is assigned as the prediction.

4.  **Evolution**: The population evolves over generations using canonical
    evolutionary operators:
    -   **Selection**: Tournament selection identifies promising parent classifiers.
    -   **Crossover**: Offspring are created by swapping prototype vectors between parents.
    -   **Mutation**: Prototype vectors are perturbed with Gaussian noise to
        explore new areas of the feature space.

5.  **Termination**: The process concludes after a fixed number of generations.
    The highest-performing set of prototype vectors found during the evolution
    is saved as the final model.
-----------------------------------------------------------------------------
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable, Set

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# =============================================================================
# =============================================================================
## 1. DATA HANDLING
# =============================================================================
# =============================================================================


def load_slide_records(
    csv_path: Path, val_ids: Set[str]
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], Dict[int, int]]:
    """
    Parses a CSV file to create training and validation record lists.

    Args:
        csv_path: Path to the CSV file containing slide filenames and labels. For setup, keep the training 
        and validation slides together in one CSV, then pass the validation filenames in a separate txt file.
        
        val_ids: A set of slide stems (filenames without extension) for the validation set.

    Returns:
        A tuple containing:
        - A list of (filepath, label) tuples for training.
        - A list of (filepath, label) tuples for validation.
        - A mapping from original class labels to zero-indexed integer labels.
    """
    records: List[Tuple[str, int]] = []
    class_set: Set[int] = set()

    with open(csv_path, mode='r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        # Auto-detect common column names
        file_key = next((k for k in ['File_name', 'filename', 'slide_id'] if k in reader.fieldnames), None)
        class_key = next((k for k in ['Class', 'label'] if k in reader.fieldnames), None)

        if not file_key or not class_key:
            raise KeyError(
                f"CSV must contain a file identifier and a class label column. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            filepath = row[file_key]
            label = int(float(row[class_key]))
            records.append((filepath, label))
            class_set.add(label)

    # Map labels to a zero-indexed range (e.g., {1, 2} -> {0, 1})
    label_map = {c: i for i, c in enumerate(sorted(list(class_set)))}
    mapped_records = [(path, label_map[label]) for path, label in records]

    train_records = [r for r in mapped_records if Path(r[0]).stem not in val_ids]
    val_records = [r for r in mapped_records if Path(r[0]).stem in val_ids]

    return train_records, val_records, label_map


def load_features(
    records: List[Tuple[str, int]], features_dir: Path
) -> List[Dict[str, Any]]:
    """
    Loads pre-computed features for a list of slide records.

    This function calculates the mean feature vector for each slide, which
    serves as the slide-level representation.

    Args:
        records: A list of (filepath, label) tuples.
        features_dir: The directory containing .npz feature files.

    Returns:
        A list of dictionaries, where each dictionary represents a slide
        and contains its name, label, and average feature vector.
    """
    print(f"Loading and processing features for {len(records)} slides...")
    prepared_data: List[Dict[str, Any]] = []

    for file_path, label in tqdm(records, desc="Loading Features"):
        stem = Path(file_path).stem
        npz_path = features_dir / f"{stem}.npz"
        if not npz_path.exists():
            print(f"Warning: Feature file not found for {stem}. Skipping.")
            continue

        try:
            features = np.load(npz_path)["features"]
            if features.ndim == 2 and features.shape[0] > 0:
                avg_feature = np.mean(features, axis=0)
                prepared_data.append({
                    "full_path": file_path,
                    "name": stem,
                    "avg_feature": avg_feature,
                    "label": label
                })
            else:
                print(f"Warning: Invalid or empty feature array in {npz_path}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not load or process {npz_path}. Error: {e}. Skipping.")

    return prepared_data


# =============================================================================
# =============================================================================
# 2. EVOLUTIONARY PROTOTYPE CLASSIFIER
# =============================================================================
# =============================================================================

class EvolutionaryPrototypeClassifier:
    """
    A classifier that evolves a set of prototype vectors for WSI classification.
    """
    def __init__(self,
                 train_data: List[Dict[str, Any]],
                 val_data: List[Dict[str, Any]],
                 prototypes_per_class: int = 20,
                 population_size: int = 50,
                 generations: int = 100,
                 elitism_count: int = 2,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.05,
                 metric: str = 'cosine'):

        self.train_data = train_data
        self.val_data = val_data
        self.prototypes_per_class = prototypes_per_class
        self.feature_dim = self.train_data[0]['avg_feature'].shape[0]

        self.distance_func: Callable[[np.ndarray, np.ndarray], float]
        if metric == 'cosine':
            self.distance_func = cosine
        elif metric == 'euclidean':
            self.distance_func = euclidean
        else:
            raise ValueError(f"Unsupported metric: {metric}. Choose 'cosine' or 'euclidean'.")
        self.metric = metric

        # EA Hyperparameters
        self.population_size = population_size
        self.generations = generations
        self.elitism_count = elitism_count
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        # Internal state
        self.population: List[Dict[str, List[np.ndarray]]] = []
        self.best_model: Dict[str, List[np.ndarray]] | None = None
        self.best_fitness: float = -1.0

    def _initialize_population(self):
        """Initializes the population with prototypes sampled from training data."""
        print("Initializing population...")
        features_c0 = [d['avg_feature'] for d in self.train_data if d['label'] == 0]
        features_c1 = [d['avg_feature'] for d in self.train_data if d['label'] == 1]

        if not features_c0 or not features_c1:
            raise ValueError("Training data must contain samples from both classes (0 and 1).")

        for _ in range(self.population_size):
            individual = {'class0_prototypes': [], 'class1_prototypes': []}
            for _ in range(self.prototypes_per_class):
                individual['class0_prototypes'].append(random.choice(features_c0).copy())
                individual['class1_prototypes'].append(random.choice(features_c1).copy())
            self.population.append(individual)

    def _predict_single(self, slide_feature: np.ndarray, model: Dict[str, List[np.ndarray]]) -> int:
        """Predicts the class for a single slide's feature vector."""
        dist_c0 = min(self.distance_func(slide_feature, p) for p in model['class0_prototypes'])
        dist_c1 = min(self.distance_func(slide_feature, p) for p in model['class1_prototypes'])
        return 0 if dist_c0 <= dist_c1 else 1

    def _evaluate_fitness(self, individual: Dict[str, List[np.ndarray]]) -> float:
        """Calculates the fitness (accuracy) of a single individual."""
        eval_data = self.val_data or self.train_data
        if not eval_data:
            return 0.0

        labels = [d['label'] for d in eval_data]
        predictions = [self._predict_single(d['avg_feature'], individual) for d in eval_data]
        return accuracy_score(labels, predictions)

    def _selection(self, fitnesses: List[float]) -> Dict[str, List[np.ndarray]]:
        """Selects a parent using tournament selection."""
        tournament_size = 5
        indices = np.random.choice(self.population_size, tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in indices]
        winner_index = indices[np.argmax(tournament_fitnesses)]
        return self.population[winner_index]

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Performs uniform crossover on the prototype vectors."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = {'class0_prototypes': [], 'class1_prototypes': []}
        child2 = {'class0_prototypes': [], 'class1_prototypes': []}

        for i in range(self.prototypes_per_class):
            # Crossover for class 0 prototypes
            if random.random() < 0.5:
                child1['class0_prototypes'].append(parent1['class0_prototypes'][i])
                child2['class0_prototypes'].append(parent2['class0_prototypes'][i])
            else:
                child1['class0_prototypes'].append(parent2['class0_prototypes'][i])
                child2['class0_prototypes'].append(parent1['class0_prototypes'][i])

            # Crossover for class 1 prototypes
            if random.random() < 0.5:
                child1['class1_prototypes'].append(parent1['class1_prototypes'][i])
                child2['class1_prototypes'].append(parent2['class1_prototypes'][i])
            else:
                child1['class1_prototypes'].append(parent2['class1_prototypes'][i])
                child2['class1_prototypes'].append(parent1['class1_prototypes'][i])

        return child1, child2

    def _mutate(self, individual: Dict) -> Dict:
        """Applies Gaussian mutation to prototype vectors."""
        for key in ['class0_prototypes', 'class1_prototypes']:
            for i in range(len(individual[key])):
                if random.random() < self.mutation_rate:
                    vector = individual[key][i]
                    mutation = np.random.normal(0, self.mutation_strength, vector.shape)
                    individual[key][i] += mutation
        return individual

    def run_evolution(self):
        """Executes the main evolutionary training loop."""
        self._initialize_population()
        print(f"\n🚀 Starting Evolutionary Prototype Optimization...")

        for gen in range(self.generations):
            # 1. Evaluate Fitness
            fitnesses = [self._evaluate_fitness(ind) for ind in tqdm(self.population, desc=f"Gen {gen+1}/{self.generations}")]

            # 2. Track Best Performer
            best_idx_this_gen = np.argmax(fitnesses)
            if fitnesses[best_idx_this_gen] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx_this_gen]
                self.best_model = self.population[best_idx_this_gen].copy()

            print(f"Generation {gen+1:03d} | Best Fitness (Val Accuracy): {self.best_fitness:.4f}")

            # 3. Elitism: Carry over the best individuals
            elite_indices = np.argsort(fitnesses)[-self.elitism_count:]
            new_population = [self.population[i] for i in elite_indices]

            # 4. Generate new population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                parent1 = self._selection(fitnesses)
                parent2 = self._selection(fitnesses)
                child1, child2 = self._crossover(parent1, parent2)

                new_population.append(self._mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(child2))

            self.population = new_population

        print(f"✅ Training finished. Best validation accuracy: {self.best_fitness:.4f}")
        return self.best_model

    def save_model(self, path: Path):
        """Saves the best-found model to a JSON file."""
        if not self.best_model:
            print("Warning: No model to save. Was the training run?")
            return
        
        print(f"Saving best model to {path}...")
        model_data = {
            "feature_dim": self.feature_dim,
            "metric": self.metric,
            "prototypes": {
                "class0": [p.tolist() for p in self.best_model['class0_prototypes']],
                "class1": [p.tolist() for p in self.best_model['class1_prototypes']],
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
        print("Model saved.")

    def load_model(self, path: Path):
        """Loads a pre-trained model from a JSON file."""
        print(f"Loading model from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        self.feature_dim = model_data['feature_dim']
        self.metric = model_data.get('metric', 'cosine')  # Default to cosine
        self.distance_func = cosine if self.metric == 'cosine' else euclidean

        self.best_model = {
            'class0_prototypes': [np.array(p) for p in model_data['prototypes']['class0']],
            'class1_prototypes': [np.array(p) for p in model_data['prototypes']['class1']],
        }
        print(f"Model loaded successfully (metric: {self.metric}).")

    def predict(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generates predictions for a list of test slides."""
        if not self.best_model:
            raise RuntimeError("Model has not been trained or loaded. Cannot predict.")

        results = []
        for slide in tqdm(test_data, desc="Predicting"):
            prediction = self._predict_single(slide['avg_feature'], self.best_model)
            results.append({'File_name': slide['full_path'], 'Predicted_Class': prediction})
        return results

# =============================================================================
# =============================================================================
# 3. COMMAND-LINE INTERFACE
# =============================================================================
# =============================================================================

def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="EP-MIL: Evolutionary Prototype-based MIL for WSI Classification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # --- Common arguments for both train and predict ---
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--features_dir", type=Path, required=True,
        help="Directory containing pre-computed slide features in .npz format."
    )

    # --- Training Parser ---
    train_parser = subparsers.add_parser("train", parents=[common_parser], help="Train a new model.")
    train_parser.add_argument("--train_csv", type=Path, required=True, help="CSV file for training data.")
    train_parser.add_argument("--val_ids_txt", type=Path, help="Optional .txt file with slide IDs for validation.")
    train_parser.add_argument("--save_model_path", type=Path, default="ep_mil_model.json", help="Path to save the trained model.")
    train_parser.add_argument("--metric", type=str, default='euclidean', choices=['cosine', 'euclidean'], help="Distance metric for vector comparison.")
    # EA Hyperparameters
    ea_group = train_parser.add_argument_group('Evolutionary Algorithm Hyperparameters')
    ea_group.add_argument("--prototypes_per_class", type=int, default=10, help="Number of prototype vectors per class.")
    ea_group.add_argument("--generations", type=int, default=50, help="Number of generations for evolution.")
    ea_group.add_argument("--population_size", type=int, default=100, help="Number of individuals in the population.")
    ea_group.add_argument("--mutation_rate", type=float, default=0.2, help="Probability of mutating a prototype.")
    ea_group.add_argument("--mutation_strength", type=float, default=0.05, help="Magnitude of Gaussian mutation.")
    ea_group.add_argument("--crossover_rate", type=float, default=0.8, help="Probability of crossover between parents.")

    # --- Prediction Parser ---
    predict_parser = subparsers.add_parser("predict", parents=[common_parser], help="Predict using a trained model.")
    predict_parser.add_argument("--test_csv", type=Path, required=True, help="CSV file for test data.")
    predict_parser.add_argument("--model_checkpoint", type=Path, required=True, help="Path to the trained model checkpoint (.json).")
    predict_parser.add_argument("--output_csv", type=Path, default="predictions.csv", help="Path to save prediction results.")

    return parser

def main():
    """Main function to handle training and prediction."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    args.features_dir.mkdir(exist_ok=True)

    if args.mode == "train":
        # --- Training Mode ---
        val_ids = set(args.val_ids_txt.read_text().splitlines()) if args.val_ids_txt else set()
        train_rec, val_rec, label_map = load_slide_records(args.train_csv, val_ids)
        
        if not train_rec:
            raise ValueError("No training data loaded. Check CSV paths and validation IDs.")
            
        print(f"Loaded {len(train_rec)} training slides and {len(val_rec)} validation slides.")
        print(f"Label map created: {label_map}")

        train_data = load_features(train_rec, args.features_dir)
        val_data = load_features(val_rec, args.features_dir) if val_rec else []

        classifier = EvolutionaryPrototypeClassifier(
            train_data=train_data, val_data=val_data,
            prototypes_per_class=args.prototypes_per_class,
            population_size=args.population_size,
            generations=args.generations,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            mutation_strength=args.mutation_strength,
            metric=args.metric
        )
        classifier.run_evolution()
        
        # Save model and label map
        args.save_model_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save_model(args.save_model_path)
        
        label_map_path = args.save_model_path.with_name(f"{args.save_model_path.stem}_label_map.json")
        with open(label_map_path, "w") as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=4)
        print(f"Label map saved to {label_map_path}")

    elif args.mode == "predict":
        # --- Prediction Mode ---
        classifier = EvolutionaryPrototypeClassifier(train_data=[{'avg_feature': np.array([0]), 'label': 0}], val_data=[])
        classifier.load_model(args.model_checkpoint)

        label_map_path = args.model_checkpoint.with_name(f"{args.model_checkpoint.stem}_label_map.json")
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map not found at {label_map_path}. It must be in the same directory as the model.")
        
        with open(label_map_path) as f:
            label_map = json.load(f)
        inverse_map = {v: int(k) for k, v in label_map.items()}

        test_recs, _, _ = load_slide_records(args.test_csv, val_ids=set())
        test_data = load_features(test_recs, args.features_dir)
        
        if not test_data:
            print("No test data could be loaded. Aborting prediction.")
            return

        predictions = classifier.predict(test_data)

        # Map internal labels (0, 1) back to original labels
        for row in predictions:
            row['Predicted_Class'] = inverse_map.get(row['Predicted_Class'], -1)

        pd.DataFrame(predictions).to_csv(args.output_csv, index=False)
        print(f"✅ Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()

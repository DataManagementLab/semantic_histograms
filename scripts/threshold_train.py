from mmce.models.threshold.trainer import ThresholdLearner
import argparse


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "dataset",
        type=str,
        help="Name of dataset",
        choices=["wordnet", "artwork", "ecommerce", "wildlife"],
    )
    args = argument_parser.parse_args()
    threshold_learner = ThresholdLearner(args.dataset)
    threshold_learner.get_training_set()
    threshold_learner.train_nn()
    threshold_learner.visualize(threshold_learner.threshold_model_path, "threshold-nn")
    threshold_learner.interactive_test()

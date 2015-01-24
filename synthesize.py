import argparse
import glob
import os

import morphine


def main(args):
    feature_files = glob.glob(os.path.join(args.feature_directory, "*.npz"))
    # Sample the feature files
    (files, time_points,
        features) = morphine.sample_feature_files(feature_files,
                                                  args.num_samples)
    # Build a k-neighbors map
    connections = morphine.build_kneighbors_table(features, args.k_neighbors)

    # Generate a random acyclic walk through the graph
    path = morphine.rand_acyclic_walk(connections)
    print path

    # Extract grains for each index

    # Overlap and add

    # ...?

    # Profit!


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("sound_directory",
                        metavar="sound_directory", type=str,
                        help="Path to a collection of sound files.")
    parser.add_argument("feature_directory",
                        metavar="feature_directory", type=str,
                        help="Path to a collection of npz feature files, with "
                        "at least two keys: {time_points, features}.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the resulting ear candy.")
    parser.add_argument("--num_samples",
                        metavar="--num_samples", type=int, default=50000,
                        help="Number of feature samples to draw.")
    parser.add_argument("--k_neighbors",
                        metavar="--k_neighbors", type=int, default=10,
                        help="Number of neighbors to link in the graph.")
    main(parser.parse_args())

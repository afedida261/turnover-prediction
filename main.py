import argparse

from src.config import set_seed
from src.datasets import discover_dataset_specs
from src.training import (
    run_filex_experiments,
    run_imbalance_experiments,
    run_time_history_experiments,
    run_train_all_masked_experiments,
)


def main():
    parser = argparse.ArgumentParser(description="Run row-level turnover experiments on data/file1.xlsx-file3.xlsx.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--list", action="store_true", help="List discovered data/file*.xlsx files and exit.")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.list:
        specs, skipped = discover_dataset_specs(include_root_excels=False)
        for spec in specs:
            print(f"{spec.tag}: {spec.path}")
        if skipped:
            print("Skipped datasets:")
            for row in skipped:
                print(f"  {row['Dataset']}: {row['Reason']}")
        return

    result = run_filex_experiments(seed=args.seed, output_dir=args.output_dir)
    print(f"Report written to {result['output_path']}")
    print(f"Raw columns identical across all files: {result['schema']['raw_same']}")
    print(f"Modeling feature columns identical across all files: {result['schema']['feature_same']}")
    for experiment in result["results"]:
        top = experiment["model_comparison"].iloc[0]
        print(
            f"{experiment['name']}: best={experiment['best_model_name']} "
            f"Val AUC={top['Val_AUC']:.4f}, Test AUC={top['Test_AUC']:.4f}"
        )

    masked_result = run_train_all_masked_experiments(result, seed=args.seed, output_dir=args.output_dir)
    print(f"Train-all/masked-eval report written to {masked_result['output_path']}")
    for experiment in masked_result["results"]:
        top = experiment["model_comparison"].iloc[0]
        print(
            f"{experiment['name']}: best={experiment['best_model_name']} "
            f"Val AUC={top['Val_AUC']:.4f}, Test AUC={top['Test_AUC']:.4f}"
        )

    imbalance_result = run_imbalance_experiments(result, seed=args.seed, output_dir=args.output_dir)
    print(f"Imbalance report written to {imbalance_result['output_path']}")
    for _, row in imbalance_result["comparison"].iterrows():
        if row["Variant"] == "baseline":
            continue
        print(
            f"{row['Experiment']} [{row['Variant']}]: best={row['Best_Model']} "
            f"Test AUC={row['Test_AUC']:.4f}, "
            f"Recall@Top20%={row['Test_Recall@Top20%']:.4f}, "
            f"Precision@Top20%={row['Test_Precision@Top20%']:.4f}"
        )

    time_history_result = run_time_history_experiments(result, seed=args.seed, output_dir=args.output_dir)
    print(f"Time-history report written to {time_history_result['output_path']}")
    for _, row in time_history_result["comparison"].iterrows():
        if row["Variant"] == "raw_baseline":
            continue
        print(
            f"{row['Experiment']} [{row['Variant']}]: best={row['Best_Model']} "
            f"Test AUC={row['Test_AUC']:.4f}, "
            f"Recall@Top20%={row['Test_Recall@Top20%']:.4f}, "
            f"Precision@Top20%={row['Test_Precision@Top20%']:.4f}"
        )


if __name__ == "__main__":
    main()

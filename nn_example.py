from neural_netowrk_helpers import (
    generate_synthetic_data,
    get_data_loaders_reg,
    plot_sample_data,
    resample_and_interpolate,
    split_data,
    normalize_data,
    plot_training_history,
    RNNModel,
)

from train_evalutate import train_model, evaluate_model


# =====================================
# MAIN EXECUTION
# =====================================


def main():
    # Generate and prepare data
    df = generate_synthetic_data()

    # Step 0: Plot sample data
    plot_sample_data(df)

    # Step 1: Data Cleaning & Preparation Resample and interpolate
    df = resample_and_interpolate(df)

    # Step 2: Split data
    train_df, val_df, test_df = split_data(df)

    # Step 3-4: Normalize data usiing z-score normalization
    train_df_norm, val_df_norm, test_df_norm, scaler_X, scaler_y_reg = normalize_data(
        train_df, val_df, test_df
    )

    # Create datasets and data loaders
    sequence_length = 12  # 24 hours of history our first hyper param

    train_loader_reg, val_loader_reg, test_loader_reg = get_data_loaders_reg(
        train_df_norm, val_df_norm, test_df_norm, sequence_length, batch_size=2048
    )

    print(f"\nDataset sizes:")
    print(
        f"Regression - Train: {len(train_loader_reg)}, Val: {len(val_loader_reg)}, Test: {len(test_loader_reg)}"
    )

    # =====================================
    # REGRESSION TASK
    # =====================================

    print("REGRESSION TASK: Predicting Signal Power")

    # Step 5: Create and train regression model
    model_reg = RNNModel(input_size=7, hidden_size=64, num_layers=2, task="regression")
    model_reg, train_losses_reg, val_losses_reg = train_model(
        model_reg,
        train_loader_reg,
        val_loader_reg,
        num_epochs=100,
        learning_rate=0.001,
        task="regression",
    )

    # Step 6: Plot training history
    plot_training_history(train_losses_reg, val_losses_reg, "regression")

    # Step 7: Evaluate on test set
    pred_reg, target_reg = evaluate_model(
        model_reg, test_loader_reg, "regression", scaler_y_reg
    )


if __name__ == "__main__":
    results = main()


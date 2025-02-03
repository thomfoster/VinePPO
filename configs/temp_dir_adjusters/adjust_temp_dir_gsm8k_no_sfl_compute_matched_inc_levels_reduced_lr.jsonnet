{
    trainer+: {
        temp_checkpoint_dir: "temp_ppo_checkpoints_gsm8k_no_sfl_compute_matched_inc_levels_reduced_lr",
        general_training_args+: {
            // was previously 1e-6
            learning_rate: 2.5e-7,
        },
    },
    episode_generator+: {
        // num_dataset_samples_per_iteration
        dataset_num_samples_per_iteration: 256,
        inference_strategy+: {
            // num_rollouts_per_sample
            samples: 8,
        }
    },
    // num_episodes_per_iteration
    num_episodes_per_iteration: 2048,
}
{
    trainer+: {
        temp_checkpoint_dir: "temp_ppo_checkpoints_math_no_sfl_compute_matched"
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
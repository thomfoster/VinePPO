{
    trainer+: {
        temp_checkpoint_dir: "temp_ppo_checkpoints_gsm8k_no_sfl_compute_matched_inc_rollouts"
    },
    episode_generator+: {
        // num_dataset_samples_per_iteration
        dataset_num_samples_per_iteration: 64,
        inference_strategy+: {
            // num_rollouts_per_sample
            samples: 32,
        }
    },
    // num_episodes_per_iteration
    num_episodes_per_iteration: 2048,
}
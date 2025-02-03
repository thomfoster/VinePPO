local sfl_n_to_k_ratio = 4;
local num_dataset_samples_per_iteration = 64;

{
    trainer+: {
        temp_checkpoint_dir: "temp_ppo_checkpoints_true_sfl_check_p0"
    },
    episode_generator+: {
        sfl_enabled: true,
        sfl_n_to_k_ratio: sfl_n_to_k_ratio,
        dataset_num_samples_per_iteration: num_dataset_samples_per_iteration * sfl_n_to_k_ratio,
    },
}
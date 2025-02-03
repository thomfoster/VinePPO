{
    trainer+: {
        temp_checkpoint_dir: "temp_ppo_checkpoints_true_sfl_very_fast_config_math_p0.1"
    },
    episode_generator+: {
        type: 'math_episode_generator_w_sfl',
        T: 5,
        N: 128,
        K: 64,
        L: 8,
        N_l: 64,
        p: 0.1,
    },
}
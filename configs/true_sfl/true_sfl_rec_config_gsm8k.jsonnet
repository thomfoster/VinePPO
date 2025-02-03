{
    trainer+: {
        temp_checkpoint_dir: "temp_ppo_checkpoints_true_sfl_rec_config_gsm8k"
    },
    episode_generator+: {
        type: 'math_episode_generator_w_sfl',
        T: 50,
        N: 1280,
        K: 256,
        L: 8,
        N_l: 64,
        p: 0.5,
    },
}
{
    trainer+: {
        temp_checkpoint_dir: "temp_ppo_checkpoints_true_sfl_check_p0"
    },
    episode_generator+: {
        type: 'math_episode_generator_w_sfl',
        T: 50,
        N: 64,
        K: 64,
        L: 8,
        N_l: 64,
        p: 0.0,
    },
}
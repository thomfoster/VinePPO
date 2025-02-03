{
    num_iterations: 1000*3,
    trainer+: {
        general_training_args+: {
            warmup_ratio: 0.03 / 3,
        },
    }
}
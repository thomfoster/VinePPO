local hf_model_name = 'realtreetune/rho-1b-sft-GSM8K';
local task = (import 'tasks/gsm8k_orig_format.jsonnet');
local total_num_iterations = 675;


(import 'polIter_rho1bSft2_ppo_MATH.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: task,
        reward_function+: { math_task: $.episode_generator.task },

        initial_model_name_or_path: hf_model_name,

        inference_strategy+: {
            guidance_llm: (import 'guidance_llms/rho1b-sft-GSM8K.jsonnet') + { api_base: 'none' },
        },
    },
    num_iterations: total_num_iterations,
}
+ (import 'sft_rho1b_for_gsm8k_eval.jsonnet')
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')

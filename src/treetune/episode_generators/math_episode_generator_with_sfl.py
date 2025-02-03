import json
import uuid
from typing import Any, Dict, List, Union, Optional, Tuple

import evaluate
import numpy as np
from datasets import Dataset

from treetune.episode_generators import EpisodeGenerator
from treetune.episode_generators.base_episode_generator import Episode
from treetune.episode_generators.episode_generator_with_reward_function import (
    EpisodeGeneratorWithRewardFunction,
    RewardFunction,
)
from treetune.logging_utils import get_logger
from treetune.tasks import Task, GSM8K
from treetune.tasks.math import MATH
from treetune.tokenization_utils import Tokenizer

import copy
import json
import logging
import random
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple, Callable

import torch.cuda
from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets

from treetune.common import Lazy
from treetune.common.gpu_utils import get_gpu_memory, wait_for_memory_release
from treetune.common.py_utils import find_n_free_ports
from treetune.common.vllm_server import VLLMServer, compute_vllm_stats
from treetune.episode_generators.base_episode_generator import EpisodeGenerator, Episode
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.episode_generators.math_episode_generator import MathEpisodeGenerator
from treetune.logging_utils import get_logger
from treetune.tasks.base_task import Task

logger = get_logger(__name__)


@EpisodeGenerator.register("math_episode_generator_w_sfl")
class MathEpisodeGeneratorWithSFL(MathEpisodeGenerator):
    def __init__(
        self,
        T: int,
        N: int,
        K: int,
        L: int,
        N_l: int,
        p: float,
        **kwargs,
    ):
        """
        # T = 50 # Number of iterations before refreshing buffer  
        # N = 1280 # Number of candidate levels to select buffer from
        # K = 256 # Buffer size
        # L = 8 # max rollouts to use for variance calculation
        # N_l = 64 # Number of levels to train on for each iteration, ie batch size
        # p = 0.5 # Proportion of current iteration train levels to sample from buffer vs sample randomly
        """
        super().__init__(**kwargs)
        self.T = T
        self.N = N
        self.K = K
        self.L = L
        self.N_l = N_l
        self.p = p
        self._logger = logger
        
    def generate(
        self, iteration: Optional[int] = None, latest_policy_path: Optional[Path] = None
    ):
        """
        Generate episodes by sampling from the model.
        This code is copied from OnPolicyEpisodeGenerator but with some modifications to include SFL.
        """
        
        #########################################
        # Memory Allocation Preamble
        # ---------------------------------------
        
        this_process_device = self.distributed_state.device
        release_memory()
        
        if self.vllm_min_available_gpu_memory_mb is not None:
            total_mem_mb = (
                torch.cuda.get_device_properties(this_process_device.index).total_memory
                / 1024**2
            )
            used_threshold_mb = total_mem_mb - self.vllm_min_available_gpu_memory_mb
            logger.info(
                f"Need at least {self.vllm_min_available_gpu_memory_mb}. "
                f"Waiting for GPU{this_process_device.index} used memory to be below {used_threshold_mb} MB. "
                f"Total GPU memory: {total_mem_mb} MB."
            )
            wait_for_memory_release(
                this_process_device.index,
                threshold_mb=used_threshold_mb,
            )
        
        gpu_memory_usage_before_mb = get_gpu_memory()[this_process_device.index]
        
        from deepspeed.runtime.utils import see_memory_usage
        
        see_memory_usage(f"Before generating episodes", force=True)
        
        # ---------------------------------------
        # end of memory allocation preamble
        #########################################
        
        if iteration is None:
            self._log_on_main(
                "Iteration is None. Using 0 as the iteration.", level="warning"
            )
            iteration = 0
            
        process_index = self.distributed_state.process_index
        
        temp_dir = self.temp_dir_root / f"iteration__{iteration:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        if latest_policy_path is None:
            hf_ckpt_path_or_model = self.initial_model_name_or_path
        else:
            hf_ckpt_path_or_model = str(latest_policy_path)
        
        #########################################
        # SFL Implementation
        # ---------------------------------------
        
        # Hyperparameters
        T = self.T
        N = self.N
        K = self.K
        L = self.L
        N_l = self.N_l
        p = self.p

        seed_buffer_candidate_sampling = self.seed + iteration
        seed_buffer_inference = self.seed + process_index * 1000 + iteration
        seed_batch_buffer_selection = self.seed + process_index * 1000 + iteration
        seed_batch_new_level_sampling = (1000*self.seed)%123456 + process_index * 1000 + iteration # should be different to buffer candidate sampling
        seed_batch_inference = (1000*self.seed)%123456 + process_index * 1000 + iteration
        seed_episode_concatenation = self.seed + iteration
                
        def randomly_sample_n_new_problems(n, seed):
            if self._orig_ds is None:
                with self.distributed_state.main_process_first():
                    self._init_orig_ds()
                    
            dataset = self._orig_ds
            dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(range(n))
                
            return dataset
        
        if iteration % T == 0:
            # get_new_buffer()
            # sample n new levels 
            seed = seed_buffer_candidate_sampling
            candidate_levels = randomly_sample_n_new_problems(N, seed)
            self._log_on_main(f"Candidate levels size(portion={self.dataset_portion}): {len(candidate_levels)}")
            # save them to disk
            inp_ds_path = temp_dir / f"candidate_levels__{process_index}"
            candidate_levels.save_to_disk(inp_ds_path)
            del candidate_levels
            # load them again
            candidate_levels = Dataset.load_from_disk(str(inp_ds_path))
            # shard the dataset
            candidate_levels = candidate_levels.shard(
                num_shards=self.distributed_state.num_processes,
                index=process_index,
                contiguous=True,
            )
            # run inference on the shard
            results_dir = temp_dir / "infer_results" / f"candidate_levels__{process_index}"
            results_dir.mkdir(parents=True, exist_ok=True)
            seed = seed_buffer_inference
            vllm_init_fn = self._get_vllm_init_fn(
                results_dir=results_dir,
                hf_ckpt_path_or_model=hf_ckpt_path_or_model,
                process_index=process_index,
                seed=seed
            )
            
            metrics = {}
            t0= time.time()
            
            def vllm_cleanup_fn():
                if self.wait_until_memory_release:
                    threshold_mb = (
                        gpu_memory_usage_before_mb * 1.1
                    ) # allow for 10% tolerance
                    wait_for_memory_release(
                        this_process_device.index,
                        threshold_mb=threshold_mb,
                    )
            
            logger.info(f"Process {process_index} starting inference on candidate levels")
            infer_results = self._run_inference(
                dataset_shard=candidate_levels,
                vllm_init_fn=vllm_init_fn,
                vllm_cleanup_fn=vllm_cleanup_fn,
                results_root_dir=results_dir,
                seed=seed,
                iteration=iteration
            )
            
            metrics["timing/episode_generation/candidate_inference"] = time.time() - t0
            
            logger.info(f"Process {process_index} finished inference on candidate levels")
            
            # Call self._generate_episodes(inference_results, iteration)
            episodes_groups, group_problem_ids, metrics_groups = self._generate_episodes(infer_results, iteration)
            
            # Generate episodes isn't doing sfl so we need to do that here
            # Generate episodes also isn't going to be logging so we need to do that here
            selected_episodes, selected_group_problem_ids = self._select_episodes(
                iteration,
                episodes_groups, 
                group_problem_ids, 
                metrics_groups,
                keep_fraction=K/N,
                max_rollouts_to_use_for_variance=L,
                logging_prefix="buffer_creation/"
            )
            
            # Map the selected episodes back to the levels from the dataset, forming the buffer
            selected_levels = candidate_levels.filter(lambda x: x["_treetune__idx"] in selected_group_problem_ids)
            print(f"Selected levels size after filtering: {len(selected_levels)}")
            logger.warning(f"Selected levels size after filtering: {len(selected_levels)}")

            # save to disk
            selected_levels.save_to_disk(temp_dir / f"buffer__{process_index}")
            logger.info(f"Buffer saved to disk at {temp_dir / f'buffer__{process_index}'}")
            
            # del unused variables
            del candidate_levels
            del selected_levels
            del infer_results
            
            self.distributed_state.wait_for_everyone()

        # get_batch()
        # load buffer from disk (for iterations not divisible by T this will be the previous buffer)
        most_recent_buffer_generation_iteration = (iteration // T) * T
        most_recent_buffer_temp_dir = self.temp_dir_root / f"iteration__{most_recent_buffer_generation_iteration:04d}"
        buffer_path = most_recent_buffer_temp_dir / f"buffer__{process_index}"
        logger.info("Loading buffer from disk from path: " + str(buffer_path))
        buffer = Dataset.load_from_disk(str(buffer_path))
        
        # select p*N_l levels from the buffer
        seed = seed_batch_buffer_selection
        buffer = buffer.shuffle(seed=seed)
        buffer = buffer.select(range(int(p*N_l)))
        
        # get (1-p)*N_l new levels
        seed = seed_batch_new_level_sampling
        new_levels = randomly_sample_n_new_problems(int((1-p)*N_l), seed)
        
        # combine together
        merged = concatenate_datasets([buffer, new_levels])
        
        # save to disk as input_dataset__{process_index} ie how it was before
        merged.save_to_disk(temp_dir / f"input_dataset__{process_index}")
            
        # TODO: Could run buffer selection and new levels sepereately to allow for individual logging
        # run_inference as normal
        # code from here should be identical to the original generate method, bar seeding
        results_dir = temp_dir / "infer_results" / f"process_{process_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        seed = seed_batch_inference
        vllm_init_fn = self._get_vllm_init_fn(
            results_dir=results_dir,
            hf_ckpt_path_or_model=hf_ckpt_path_or_model,
            process_index=process_index,
            seed=seed
        )
        
        metrics = {}
        t0= time.time()
        
        def vllm_cleanup_fn():
            if self.wait_until_memory_release:
                threshold_mb = (
                    gpu_memory_usage_before_mb * 1.1
                ) # allow for 10% tolerance
                wait_for_memory_release(
                    this_process_device.index,
                    threshold_mb=threshold_mb,
                )
                
        logger.info(f"Process {process_index} starting inference")
        infer_results = self._run_inference(
            dataset_shard=merged,
            vllm_init_fn=vllm_init_fn,
            vllm_cleanup_fn=vllm_cleanup_fn,
            results_root_dir=results_dir,
            seed=seed,
            iteration=iteration
        )
        
        metrics["timing/episode_generation/inference"] = time.time() - t0
        
        logger.info(f"Process {process_index} finished inference")
        
        #########################################
        # Handle inference results
        # ---------------------------------------
        
        t0 = time.time()
        
        # Generate episodes from the inference results. Each process generates its own episodes.
        logger.info(f"Process {process_index} starting episode generation")
        episodes_groups, group_problem_ids, metrics_groups = self._generate_episodes(infer_results, iteration)
        
        # flatten groyps, ids, and metrics
        episodes = [episode for group in episodes_groups for episode in group]
        problem_ids = [[id]*len(group) for id, group in zip(group_problem_ids, episodes_groups)]
        metrics = {}
        for metrics in metrics_groups:
            for key in metrics:
                metrics.setdefault(key, []).extend(metrics[key])
        
        self.process_metrics(metrics, iteration, parent_prefix="", prefix=f"")
        
        # TODO: Seperate out the metrics from levels in buffer vs those from new levels
        # we know this information from the ids and buffer dataset
        # we can then log them seperately
        
        
        # convert episodes to dicts
        episodes_lst = [
            self._convert_to_dict(e)
            for e in episodes
        ]
        episodes_ds_shard = Dataset.from_list(episodes_lst)
        episodes_ds_shard.save_to_disk(
            temp_dir / f"episodes" / f"shard_{process_index:02d}"
        )
        del episodes_ds_shard
        release_memory()
        metrics["timing/episode_generation/inferResult_to_episodes"] = time.time() - t0
        
        # log the vLLM stats
        if self.distributed_state.is_main_process:
            try:
                vllm_stats = compute_vllm_stats(results_dir / "vllm_server.log")
            except Exception as e:
                logger.error(f"Error while computing vLLM stats: {e}")
                vllm_stats = {}
            
            if "avg_generation_throughput" in vllm_stats:
                vllm_stats["total_approx_generation_throughput"] = (
                    vllm_stats["avg_generation_throughput"]
                    * self.distributed_state.num_processes
                )
                
            # round to 2 decimal places
            vllm_stats = {f"vllm_stats/{k}": round(v, 2) for k,v in vllm_stats.items()}
            logger.info(f"vLLM Stats: {vllm_stats}")
            metrics.update(vllm_stats)
        
        self._cloud_log(metrics)
        
        # Concatenate all episodes shards
        self.distributed_state.wait_for_everyone()
        seed = seed_episode_concatenation
        if self.is_main_process():
            shard_paths = list((temp_dir / f"episodes").glob("shard_*"))
            shard_paths.sort(key=lambda x: int(x.name.split("shard_")[-1]))

            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
            )
            if self.num_episodes_per_iteration is None:
                pass
            elif len(merged) > self.num_episodes_per_iteration:
                merged = merged.shuffle(seed=seed)
                merged = merged.select(range(self.num_episodes_per_iteration))
            elif len(merged) < self.num_episodes_per_iteration:
                if self.fill_missing_episodes:
                    # Fill the missing episodes by repeating the existing ones
                    logger.warning(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration}). "
                        f"Repeating the existing episodes."
                    )
                    num_repeats = self.num_episodes_per_iteration // len(merged) + 1
                    merged = concatenate_datasets([merged] * num_repeats)
                    merged = merged.shuffle(seed=seed)
                    merged = merged.select(range(self.num_episodes_per_iteration))
                    logs = {f"episodes_metric/fill_missing_episodes": num_repeats}
                    self._cloud_log({**logs, "train/global_iteration": iteration})
                else:
                    raise ValueError(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration})"
                    )

            merged.save_to_disk(temp_dir / "episodes" / "merged")
            del merged
            release_memory()
            
        self.distributed_state.wait_for_everyone()
        episodes = Dataset.load_from_disk(str(temp_dir / "episodes" / "merged"))
        
        see_memory_usage(f"After generating episodes", force=True)
        
        logger.info(f"Process {process_index} finished .generate and has {len(episodes)} episodes")
        
        self._save_generations_to_cloud(temp_dir, iteration)
        self._clean_up_temp_dir(temp_dir)

        self.distributed_state.wait_for_everyone()
        
        # ---------------------------------------
        # end of handling inference results
        #########################################
    
        return episodes
    
    def process_metrics(self, metrics, iteration, parent_prefix="", prefix=""):
        
        if "is_unfinished_response" in metrics:
            metrics["is_unfinished_response"] = sum(
                metrics["is_unfinished_response"]
            ) / len(metrics["is_unfinished_response"])

        if "empty_response" in metrics:
            metrics["empty_response"] = sum(metrics["empty_response"]) / len(
                metrics["empty_response"]
            )

        if "num_reasoning_steps" in metrics:
            num_reasoning_steps = np.array(metrics.pop("num_reasoning_steps"))
            metrics["num_reasoning_steps/dist"] = num_reasoning_steps
            metrics["num_reasoning_steps/mean"] = np.mean(num_reasoning_steps)

        if "parse_failed" in metrics:
            metrics["parse_failed"] = sum(metrics["parse_failed"]) / len(
                metrics["parse_failed"]
            )

        if "once_hit" in metrics:
            metrics["once_hit"] = sum(metrics["once_hit"]) / len(metrics["once_hit"])
            
            
        for k in metrics:
            if isinstance(k, str) and k.startswith("hit_") and k.endswith("_times"):
                logger.info(f"Averaging out {k}")
                metrics[k] = sum(metrics[k]) / len(metrics[k])

        if "hit_variance" in metrics:
            metrics["hit_variance"] = sum(metrics["hit_variance"]) / len(
                metrics["hit_variance"]
            )
            
        if "n_hits" in metrics:
            metrics["n_hits"] = sum(metrics["n_hits"]) / len(metrics["n_hits"])
            
        if "avg_hits" in metrics:
            metrics["avg_hits"] = sum(metrics["avg_hits"]) / len(metrics["avg_hits"])

        if "trajectory_bleu" in metrics:
            metrics["trajectory_bleu"] = sum(metrics["trajectory_bleu"]) / len(
                metrics["trajectory_bleu"]
            )

        if len(metrics) > 0:
            logs = {f"episodes_metric/{parent_prefix}{prefix}{k}": v for k, v in metrics.items()}
            self._cloud_log({**logs, "train/global_iteration": iteration})

    def _select_episodes(
        self, 
        iteration: int,
        episodes_groups: List[List[Episode]], 
        group_problem_ids: List[List[int]], 
        metrics_groups: List[Dict[str, Any]],
        keep_fraction: float,
        max_rollouts_to_use_for_variance: int,
        logging_prefix: Optional[str],
    ) -> Tuple[List[Episode], List[int]]:
        """
        Selects episodes based on the SFL algorithm.
        Also does some logging of pre/post selection metrics.
        This code is copied from MathEpisodeGenerator but with some modifications to include SFL.
        """
        
        # first flatten metrics groups and log pre selection
        all_metrics = {}
        for metrics in metrics_groups:
            for key in metrics:
                all_metrics.setdefault(key, []).extend(metrics[key])
        self.process_metrics(all_metrics, iteration, parent_prefix=logging_prefix, prefix="")

        # find the top 1/4 of episodes groups based on variance of scores        
        logger.info("~"*80)
        scores = [[e.scores for e in episode_group] for episode_group in episodes_groups]
        scores = [group_scores[:max_rollouts_to_use_for_variance] for group_scores in scores]
        variances = [np.var(score) for score in scores]
        sorted_indices = np.argsort(variances)[::-1]
        
        if keep_fraction <= 1.0:
            logger.info(f"Doing SFL selection on {len(episodes_groups)} episodes groups")
            n_to_keep = int(len(sorted_indices) * keep_fraction)
            selected_indices = sorted_indices[:n_to_keep]
        else:
            logger.info(f"WARNING: No SFL {keep_fraction=} and not cutting down samples!!!!!!!!!")        
            selected_indices = range(len(sorted_indices))
        
        # perform selection and flatten groups
        selected_episodes_groups = [episodes_groups[i] for i in selected_indices]
        selected_metrics_groups = [metrics_groups[i] for i in selected_indices]
        selected_group_problem_ids = [group_problem_ids[i] for i in selected_indices]
        selected_episodes = [episode for group in selected_episodes_groups for episode in group]
        logger.info(f"Took {len(episodes_groups)} episodes groups and selected {len(selected_episodes_groups)} groups, which is {len(selected_episodes)} episodes")
        selected_metrics = {}
        for metrics in selected_metrics_groups:
            for key in metrics:
                selected_metrics.setdefault(key, []).extend(metrics[key])
        
        self.process_metrics(selected_metrics, iteration, parent_prefix=logging_prefix, prefix="selected/")
        
        return selected_episodes, selected_group_problem_ids
    
    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]: # TF NOTE: This is copied from MathEpisodeGenerator but don't think the type is correct
        """
        Turns the inference results into episodes.
        This code is copied from MathEpisodeGenerator but with some modifications to include SFL.
        # TODO: Logging will either need to be moved out of this method or hidden behind a flag.
        # TODO: SFL selection will either need to be moved out of this method or hidden behind a flag.
        """
        # episodes_groups, group_problem_ids, metrics_groups = self._generate_episodes(infer_results, iteration)
        # episodes, episode_problem_ids = self._generate_episodes(infer_results, iteration)
        print("INSIDE SFL EPISODE GENERATOR _generate_episodes", '#' * 100)
        episodes_groups = []
        group_problem_ids = []
        metrics_groups = []
        for idx, instance in enumerate(inference_results):
            tree = json.loads(instance["_treetune__reasoning_tree"])
            episodes = []
            metrics = {}
            paths = self.extract_paths_from_tree(tree)
            all_rewards = []
            all_responses = []
            for path in paths:
                # noinspection DuplicatedCode
                assert len(path["node_chain"]) == 2, "Does not support multi-hop paths."

                finish_reason = path["node_chain"][-1]["finish_reason"]
                query_text = path["node_chain"][0]["text"]
                full_text = path["node_chain"][-1]["full_text"]
                response_text = full_text[len(query_text) :]

                try:
                    num_reasoning_steps = self.compute_number_of_reasoning_steps(
                        response_text
                    )
                    metrics.setdefault("num_reasoning_steps", []).append(
                        num_reasoning_steps
                    )
                    metrics.setdefault("parse_failed", []).append(False)
                except Exception as e:
                    logger.error(f"Parsing reasoning steps failed {e}")
                    logger.error(f"Response: `{response_text}`")
                    metrics.setdefault("parse_failed", []).append(True)

                if finish_reason != "length":
                    # Generation stopped because the model hit <eos>
                    reward, is_unfinished_response = self.reward_function(
                        query_text, response_text, instance
                    )
                else:
                    # Generation stopped because the model hit the `max_tokens` limit
                    reward = self.reward_function.get_unfinished_response_penalty()
                    is_unfinished_response = True

                try:
                    query_token_ids, response_token_ids = (
                        self._tokenize_query_and_response(
                            query_text,
                            response_text,
                            # Only append EOS token if the response is complete
                            allow_append_eos=not is_unfinished_response,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to tokenize query and response for instance {instance['_treetune__idx']}: {e}"
                    )
                    logger.error(f"Query: {query_text}")
                    logger.error(f"Response: {response_text}")
                    metrics.setdefault("empty_response", []).append(True)
                    continue

                all_responses.append(response_text)

                if self.max_sequence_length is not None:
                    seq_len = len(query_token_ids) + len(response_token_ids)
                    if seq_len > self.max_sequence_length:
                        logger.warning(
                            f"Sequence length {seq_len} is greater than "
                            f"max sequence length {self.max_sequence_length}."
                        )

                        # Truncate the response
                        response_token_ids = response_token_ids[
                            : self.max_sequence_length - len(query_token_ids)
                        ]
                        reward = self.reward_function.get_unfinished_response_penalty()
                        is_unfinished_response = True

                if len(response_token_ids) == 0:
                    logger.warning(
                        f"Response token ids are empty for instance {instance['_treetune__idx']}"
                    )
                    metrics.setdefault("empty_response", []).append(False)
                    continue

                metrics.setdefault("empty_response", []).append(False)
                metrics.setdefault("is_unfinished_response", []).append(
                    is_unfinished_response
                )

                episode = Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    scores=float(reward),
                )

                episodes.append(episode)
                all_rewards.append(float(reward))

            if len(all_rewards) > 0:
                once_hit = any([r == 1.0 for r in all_rewards])
                metrics.setdefault("once_hit", []).append(float(once_hit))
                
            for i in range(len(paths)+1):
                n_hits = sum([r == 1.0 for r in all_rewards[:i]])
                hit_i_times = n_hits == i
                metrics.setdefault(f"hit_{i}_times", []).append(float(hit_i_times))
                
            if len(all_rewards) > 0:
                hit_variance = np.var(all_rewards)
                metrics.setdefault("hit_variance", []).append(hit_variance)
                
            if len(all_rewards) > 0:
                n_hits = sum([r == 1.0 for r in all_rewards])
                metrics.setdefault("n_hits", []).append(n_hits)
                
            if len(all_rewards) > 0:
                avg_hits = sum([r == 1.0 for r in all_rewards]) / len(all_rewards)
                metrics.setdefault("avg_hits", []).append(avg_hits)
                
            if len(all_responses) > 1:
                metrics.setdefault("num_unique_responses", []).append(
                    len(set(all_responses))
                )
                if self._bleu_metric is not None:
                    bleu = self._avg_bleu_of_pairs_of_response(all_responses)
                    metrics.setdefault("trajectory_bleu", []).append(bleu)
                    
            episodes_groups.append(episodes)
            group_problem_ids.append(instance["_treetune__idx"])
            metrics_groups.append(metrics)
            
        # log some length info
        logger.info(f"Epsiodes groups length: {len(episodes_groups)}")
        logger.info(f"Group problem ids length: {len(group_problem_ids)}")
        logger.info(f"Metrics groups length: {len(metrics_groups)}")
        
        return episodes_groups, group_problem_ids, metrics_groups
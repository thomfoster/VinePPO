import copy
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Callable, List, Optional, Union

import numpy as np
from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from treetune.common import Lazy
from treetune.common.vllm_server import VLLMServer
from treetune.episode_generators import EpisodeGenerator, MathEpisodeGenerator
from treetune.episode_generators.base_episode_generator import Episode
from treetune.inference_strategies import InferenceStrategy
from treetune.logging_utils import get_logger

logger = get_logger(__name__)


@EpisodeGenerator.register("math_episode_generator_w_sfl")
class MathEpisodeGeneratorWithSFL(MathEpisodeGenerator):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._logger = logger
        
    def generate(
        self, iteration: Optional[int] = None, latest_policy_path: Optional[Path] = None
    ):
        """
        Generate episodes by sampling from the model.
        This code is copied from OnPolicyEpisodeGenerator but with some modifications to include SFL.
        # TODO: Workout how to handle the extra kwargs like p, N, K etc
        """
        
        #########################################
        # Memory Allocation Preamble
        # ---------------------------------------
        
        this_process_deivce = self.distributed_state.device
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
        
        for deepspeed.runtime.utils import see_memory_usage
        
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
        
        #########################################
        # Dataset setup
        # ---------------------------------------
        
        # get the original dataset, assuming as a HF Dataset
        if self._orig_ds is None:
            with self.distributed_state.main_process_first():
                self._init_orig_ds()
                
        dataset = self._orig_ds
                
        # set the num samples we're going to get
        if self.dataset_num_samples_per_iteration is not None:
            num_samples = self.dataset_num_samples_per_iteration
        else:
            num_samples = int(self.initial_ds_after_filter_size * self.dataset_portion)
            self._log_on_main(
                f"Using {num_samples} for each iteration based on the dataset portion.")
            assert num_samples < len(dataset)
            
        # select num samples from the original dataset
        if not self.dataset_sample_with_replacement:
            # Split the dataset into portions and select one portion based on the iteration
            samples_per_iteration = (
                self.dataset_num_samples_per_iteration
                if self.dataset_num_samples_per_iteration is not None
                else int(self.initial_ds_after_filter_size * self.dataset_portion)
            )
            start_idx = samples_per_iteration * iteration
            end_idx = samples_per_iteration * iteration + num_samples
            dataset = dataset.select(range(start_idx, end_idx))
        else:
            # Shuffle the dataset so that the same dataset is not used in every iteration
            do_shuffle = (
                self.dataset_shuffle_on_each_iteration
                or self.dataset_shuffle_before_portion
            )
            if do_shuffle:
                dataset = dataset.shuffle(seed=self.seed + iteration)
            dataset = dataset.select(range(num_samples))
            
        # some logging about the dataset
        self._log_on_main(
            f"Dataset Size(portion={self.dataset_portion}): {len(dataset)}"
        )
        self._log_on_main(
            f"Dataset Examples: "
            f"{json.dumps([dataset[i] for i in range(min(2, len(dataset)))], indent=2, sort_keys=True)}"
        )
        
        # We're then going to save to disk
        temp_dir = self.temp_dir_root / f"iteration_{iteration:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to disk so that it's memory efficient. Note that this is done on all processes.
        # to avoid any issues with distributed environment and funkiness of HF Datasets.
        inp_ds_path = temp_dir / f"input_dataset__{process_index}"
        dataset.save_to_disk(inp_ds_path)
        del dataset
        
        # The same dataset is loaded on all processes
        dataset = Dataset.load_from_disk(str(inp_ds_path))
        
        # Shared the dataset based on the number of processes
        dataset = dataset.shard(
            num_shareds=self.distributed_state.num_processes,
            index=process_index,
            contiguous=True,
        )
        
        # ---------------------------------------
        # end of dataset setup
        #########################################
        
        #########################################
        # Inference
        # ---------------------------------------
        
        results_dir = temp_dir / "infer_results" / f"process_{process_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a seed based on self.seed, self.dist_state.process_index, and iteration
        seed = self.seed + process_index * 100 + iteration
        
        if latest_policy_path is None:
            hf_ckpt_path_or_model = self.initial_model_name_or_path
        else:
            hf_ckpt_path_or_model = str(latest_policy_path)
            
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
        
        logger.info("Class heirachy of self")
        logger.info(self.__class__.__mro__)
        logger.info(f"Process {process_index} starting inference")
        infer_results = self._run_inference(
            dataset_shard=dataset,
            vllm_init_fn=vllm_init_fn,
            vllm_cleanup_fn=vllm_cleanup_fn,
            results_root_dir=results_dir,
            seed=seed,
            iteration=iteration
        )
        
        metrics["timing/episode_generation/inference"] = time.time() - t0
        
        logger.info(f"Process {process_index} finished inference")
        
        # ---------------------------------------
        # end of inference
        #########################################
        
        #########################################
        # Handle inference results
        # ---------------------------------------
        
        t0 = time.time()
        
        # Generate episodes from the inference results. Each process generates its own episodes.
        logger.info(f"Process {process_index} starting episode generation")
        episodes, episode_problem_ids = self._generate_episodes(infer_results, iteration)
        
        # filter the dataset based on the episode_problem_ids
        dataset = dataset.filter(lambda x: x["_treetune__idx"] in episode_problem_ids)
        print(f"Dataset size after filtering: {len(dataset)}")
    
        # save to disk
        dataset.save_to_disk(temp_dir / f"selected_dataset__{process_index}")
        
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
                    * self.dist_state.num_processes
                )
                
            # round to 2 decimal places
            vllm_stats = {f"vllm_stats/{k}": round(v, 2) for k,v in vllm_stats.items()}
            logger.info(f"vLLM Stats: {vllm_stats}")
            metrics.update(vllm_stats)
        
        self._cloud_log(metrics)
        
        # Concatenate all episodes shards
        self.distributed_state.wait_for_everyone()
        if self.is_main_process():
            shard_paths = list((temp_dir / f"episodes").glob("shard_*"))
            shard_paths.sort(key=lambda x: int(x.name.split("shard_")[-1]))

            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
            )
            if self.num_episodes_per_iteration is None:
                pass
            elif len(merged) > self.num_episodes_per_iteration:
                merged = merged.shuffle(seed=self.seed + iteration)
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
                    merged = merged.shuffle(seed=self.seed + iteration)
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
        
        self._save_generations_to_cloud(temp_dir, iteration)
        self._clean_up_temp_dir(temp_dir)

        self.distributed_state.wait_for_everyone()
        
        # ---------------------------------------
        # end of handling inference results
        #########################################
        
        #########################################
        # SFL Implementation
        # ---------------------------------------
        
        # Hyperparameters
        T = 50 # Number of iterations before refreshing buffer
        N = 1280 # Number of candidate levels to select buffer from
        K = 256 # Buffer size
        N_l = 64 # Number of levels to train on for each iteration, ie batch size
        p = 0.5 # Proportion of current iteration train levels to sample from buffer vs sample randomly
                
        def randomly_sample_n_new_problems(n, seed):
            if self._orig_ds is None:
                with self.distributed_state.main_process_first():
                    self._init_orig_ds()
                    
            dataset = self._orig_ds
            dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(range(n))
                
            return dataset
        
        # if iteration means time to refresh buffer
            # get_new_buffer()
            
                # sample n new levels 
                # save them to disk
                # load them again
                # shard the dataset
                # run inference on the shard
                # Call self._generate_episodes(inference_results, iteration)
                # Generate episodes isn't doing sfl so we need to do that here
                # Generate episodes also isn't going to be logging so we need to do that here
                # Map the selected episodes back to the levels from the dataset, forming the buffer
                # Write the buffer to disk
                
        # get_batch()
            # load buffer from disk
            # select p*N_l levels from the buffer
            # get (1-p)*N_l new levels
            # combine together
            # save to disk as input_dataset__{process_index} ie how it was before
            
        # run_inference as normal # NOTE: Could run buffer selection and new levels sepereately to allow for individual logging
        # generate_episodes as normal # NOTE: Again workout where logging is going to go
            
        raise NotImplementedError("This method is not implemented for this class")
        return episodes
    
    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]: # TF NOTE: This is copied from MathEpisodeGenerator but don't think the type is correct
        """
        Turns the inference results into episodes.
        This code is copied from MathEpisodeGenerator but with some modifications to include SFL.
        # TODO: Logging will either need to be moved out of this method or hidden behind a flag.
        # TODO: SFL selection will either need to be moved out of this method or hidden behind a flag.
        """
        raise NotImplementedError("This method is not implemented for this class")
        return episodes
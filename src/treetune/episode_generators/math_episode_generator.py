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

logger = get_logger(__name__)


@RewardFunction.register("math_reward_function")
class MATHRewardFunction(RewardFunction):
    def __init__(
        self,
        tokenizer: Tokenizer,
        math_task: Task,
        penalize_unfinished_response: bool = False,
        unfinished_response_penalty: float = -1.0,
        timeout: Optional[int] = None,
    ):
        assert isinstance(math_task, (MATH, GSM8K))
        self.tokenizer = tokenizer
        self.math_task = math_task
        self.penalize_unfinished_response = penalize_unfinished_response
        self.unfinished_response_penalty = unfinished_response_penalty
        self.timeout = timeout

    def get_unfinished_response_penalty(self) -> float:
        return float(self.unfinished_response_penalty)

    def __call__(
        self, query: str, response: str, dataset_instance: Dict[str, Any]
    ) -> Tuple[float, bool]:
        pred_answer = self.math_task.extract_predicted_answer_from_text(
            response, dataset_instance["problem"]
        )
        is_unfinished_response = pred_answer is None
        if is_unfinished_response and self.penalize_unfinished_response:
            return float(self.unfinished_response_penalty), is_unfinished_response

        gold_answer = dataset_instance["answer"]
        reward = self.math_task.grade_answer(
            given_answer=pred_answer,
            ground_truth=gold_answer,
            item=dataset_instance,
            timeout=self.timeout,
        )

        return float(reward), is_unfinished_response

    def is_unfinished_response(
        self, response: str, dataset_instance: Dict[str, Any]
    ) -> bool:
        pred_answer = self.math_task.extract_predicted_answer_from_text(
            response, dataset_instance["problem"]
        )
        return pred_answer is None


@EpisodeGenerator.register("math_episode_generator")
class MathEpisodeGenerator(EpisodeGeneratorWithRewardFunction):
    def __init__(
        self,
        reasoning_step_delimiter: Optional[str] = None,
        answer_prefix: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        sfl_enabled: bool = False,
        sfl_n_to_k_ratio: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.reasoning_step_delimiter = reasoning_step_delimiter
        self.answer_prefix = answer_prefix
        self.sfl_enabled = sfl_enabled
        self.sfl_n_to_k_ratio = sfl_n_to_k_ratio

        try:
            self._bleu_metric = evaluate.load(
                "bleu",
                experiment_id=uuid.uuid4().hex,
            )
        except Exception as e:
            logger.error(f"Failed to load BLEU metric: {e}")
            self._bleu_metric = None

        assert hasattr(
            self.task, "split_solution_into_intermediate_steps"
        ), f"Task {self.task} does not have a method `split_solution_into_intermediate_steps`."

    def compute_number_of_reasoning_steps(self, response_text: str) -> int:
        # noinspection PyUnresolvedReferences
        indices = self.task.split_solution_into_intermediate_steps(response_text)
        return len(indices) - 1

    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        print("INSIDE MATH EPISODE GENERATOR", '#' * 100)
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
                    metrics.setdefault("empty_response", []).append(True)
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

            # group level metrics
            once_hit = any([r == 1.0 for r in all_rewards])
            metrics.setdefault("once_hit", []).append(float(once_hit))
                
            for i in range(len(paths)+1):
                n_hits = sum([r == 1.0 for r in all_rewards[:i]])
                hit_i_times = n_hits == i
                metrics.setdefault(f"hit_{i}_times", []).append(float(hit_i_times))
                
            hit_variance = np.var(all_rewards) if len(all_rewards) > 1 else 0.0
            metrics.setdefault("hit_variance", []).append(hit_variance)
                
            n_hits = sum([r == 1.0 for r in all_rewards])
            metrics.setdefault("n_hits", []).append(n_hits)
                
            avg_hits = sum([r == 1.0 for r in all_rewards]) / len(all_rewards)
            metrics.setdefault("avg_hits", []).append(avg_hits)
                
            metrics.setdefault("num_unique_responses", []).append(
                len(set(all_responses))
            )
            if self._bleu_metric is not None:
                if len(all_responses) > 1:
                    bleu = self._avg_bleu_of_pairs_of_response(all_responses)
                else:
                    bleu = 0.0
                metrics.setdefault("trajectory_bleu", []).append(bleu)
                    
            episodes_groups.append(episodes)
            group_problem_ids.append(instance["_treetune__idx"])
            metrics_groups.append(metrics)
            
        def process_metrics(metrics, prefix=""):
            
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
                
                
            for i in range(len(paths)+1):
                if f"hit_{i}_times" in metrics:
                    metrics[f"hit_{i}_times"] = sum(metrics[f"hit_{i}_times"]) / len(metrics[f"hit_{i}_times"])

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
                logs = {f"episodes_metric/{prefix}{k}": v for k, v in metrics.items()}
                self._cloud_log({**logs, "train/global_iteration": iteration})

        ################################################################################################
        # SFL Selection of episodes groups
        ################################################################################################
        
        # first flatten metrics groups and log pre selection
        all_metrics = {}
        for metrics in metrics_groups:
            for key in metrics:
                all_metrics.setdefault(key, []).extend(metrics[key])
        process_metrics(all_metrics, prefix="")

        # find the top 1/4 of episodes groups based on variance of scores        
        logger.info("~"*80)
        scores = [[e.scores for e in episode_group] for episode_group in episodes_groups]
        variances = [np.var(score) for score in scores]
        sorted_indices = np.argsort(variances)[::-1]
        
        if self.sfl_enabled:
            logger.info(f"Doing SFL selection on {len(episodes_groups)} episodes groups")
            selected_indices = sorted_indices[:len(sorted_indices)//self.sfl_n_to_k_ratio]
        
        else:
            # logger.info("WARNING: Not doing SFL!!!!!!!!!")
            # selected_indices = range(len(sorted_indices)//4)

            logger.info(f"Not doing SFL, selecting all {len(episodes_groups)} episodes groups")
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
        
        process_metrics(selected_metrics, prefix="selected/")
        
        return selected_episodes

    # noinspection DuplicatedCode
    def _avg_bleu_of_pairs_of_response(self, response: List[str]) -> float:
        preds = []
        refs = []
        for i in range(len(response)):
            for j in range(i + 1, len(response)):
                sen_1 = response[i]
                sen_2 = response[j]
                preds.append(sen_1)
                refs.append(sen_2)
        bleu_full_stats = self._bleu_metric.compute(predictions=preds, references=refs)
        bleu = bleu_full_stats["bleu"]
        return bleu

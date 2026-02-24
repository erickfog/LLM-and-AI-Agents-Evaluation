"""
Módulo de avaliação de LLMs e Agents usando DeepEval
"""

from .metrics_llm import (
    evaluate_answer_relevancy,
    evaluate_faithfulness,
    evaluate_bias_toxicity,
    evaluate_geval_custom,
    evaluate_llm_batch
)

from .metrics_agents import (
    evaluate_task_completion,
    evaluate_tool_correctness,
    evaluate_argument_correctness,
    evaluate_plan_quality,
    evaluate_plan_adherence,
    evaluate_step_efficiency,
    evaluate_agent_batch
)

from .utils import (
    load_llm_dataset,
    load_agent_dataset,
    format_results,
    generate_report,
    plot_metrics
)

__all__ = [
    'evaluate_answer_relevancy',
    'evaluate_faithfulness',
    'evaluate_bias_toxicity',
    'evaluate_geval_custom',
    'evaluate_llm_batch',
    'evaluate_task_completion',
    'evaluate_tool_correctness',
    'evaluate_argument_correctness',
    'evaluate_plan_quality',
    'evaluate_plan_adherence',
    'evaluate_step_efficiency',
    'evaluate_agent_batch',
    'load_llm_dataset',
    'load_agent_dataset',
    'format_results',
    'generate_report',
    'plot_metrics'
]

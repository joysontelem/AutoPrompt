from .estimator_llm import LLMEstimator


def give_estimator(opt):
    if opt.method == 'llm':
        return LLMEstimator(opt.config)
    else:
        raise ValueError(f"Unknown estimator method: {opt.method}")

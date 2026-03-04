import logging
from pathlib import Path

import yaml
from easydict import EasyDict as edict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI

LLM_ENV = yaml.safe_load(open('config/llm_env.yml', 'r'))


def get_llm(config: dict):
    """
    Returns the LLM model based on config type.
    """
    temperature = config.get('temperature', 0)
    model_kwargs = config.get('model_kwargs', {})

    if config['type'].lower() == 'openai':
        api_base = LLM_ENV['openai']['OPENAI_API_BASE'] or 'https://api.openai.com/v1'
        kwargs = dict(
            temperature=temperature,
            model_name=config['name'],
            openai_api_key=config.get('openai_api_key', LLM_ENV['openai']['OPENAI_API_KEY']),
            openai_api_base=config.get('openai_api_base', api_base),
            model_kwargs=model_kwargs,
        )
        if LLM_ENV['openai']['OPENAI_ORGANIZATION']:
            kwargs['openai_organization'] = config.get(
                'openai_organization', LLM_ENV['openai']['OPENAI_ORGANIZATION']
            )
        return ChatOpenAI(**kwargs)

    elif config['type'].lower() == 'azure':
        return AzureChatOpenAI(
            temperature=temperature,
            azure_deployment=config['name'],
            openai_api_key=config.get('openai_api_key', LLM_ENV['azure']['AZURE_OPENAI_API_KEY']),
            azure_endpoint=config.get('azure_endpoint', LLM_ENV['azure']['AZURE_OPENAI_ENDPOINT']),
            openai_api_version=config.get('openai_api_version', LLM_ENV['azure']['OPENAI_API_VERSION']),
        )

    elif config['type'].lower() == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=config['name'],
            google_api_key=LLM_ENV['google']['GOOGLE_API_KEY'],
            model_kwargs=model_kwargs,
        )

    elif config['type'].lower() == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            temperature=temperature,
            model=config['name'],
            api_key=LLM_ENV['anthropic']['ANTHROPIC_API_KEY'],
            model_kwargs=model_kwargs,
        )

    else:
        raise NotImplementedError(f"LLM type '{config['type']}' not implemented")


def load_yaml(yaml_path: str) -> edict:
    """
    Load a YAML config file and return as EasyDict.
    """
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        if 'meta_prompts' in yaml_data and 'folder' in yaml_data['meta_prompts']:
            yaml_data['meta_prompts']['folder'] = Path(yaml_data['meta_prompts']['folder'])
    return edict(yaml_data)


def load_prompt(prompt_path: str) -> PromptTemplate:
    """
    Load a prompt template from a text file.
    """
    with open(prompt_path, 'r') as file:
        prompt = file.read().rstrip()
    return PromptTemplate.from_template(prompt)

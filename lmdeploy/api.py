# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Literal, Optional, Union

from .archs import get_task, autoget_backend_config

from .messages import PytorchEngineConfig, SpeculativeConfig, TurbomindEngineConfig, TurboMindEngineConfig, DriftEngineConfig

from .model import ChatTemplateConfig





def pipeline(model_path: str,
             backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig, TurboMindEngineConfig, DriftEngineConfig]] = None,
             chat_template_config: Optional[ChatTemplateConfig] = None,
             log_level: str = 'WARNING',
             max_log_len: int = None,
             speculative_config: SpeculativeConfig = None):
    """
    Args:
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig | DriftEngineConfig): backend
            config instance. Default to None.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        log_level(str): set log level whose value among [CRITICAL, ERROR,
            WARNING, INFO, DEBUG]
        max_log_len(int): Max number of prompt characters or prompt tokens
            being printed in log

    Examples:
        >>> # LLM
        >>> import lmdeploy
        >>> pipe = lmdeploy.pipeline('internlm/internlm-chat-7b')
        >>> response = pipe(['hi','say this is a test'])
        >>> print(response)
        >>>
        >>> # VLM
        >>> from lmdeploy.vl import load_image
        >>> from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
        >>> pipe = pipeline('liuhaotian/llava-v1.5-7b',
        ...                 backend_config=TurbomindEngineConfig(session_len=8192),
        ...                 chat_template_config=ChatTemplateConfig(model_name='vicuna'))
        >>> im = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
        >>> response = pipe([('describe this image', [im])])
        >>> print(response)
    """ # noqa E501
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    from lmdeploy.utils import get_logger, get_model
    logger = get_logger('lmdeploy')
    logger.setLevel(log_level)

    _model_path = model_path
    _log_level = log_level
    _speculative_config = speculative_config
    _backend_config_for_engine = backend_config

    if isinstance(backend_config, (TurboMindEngineConfig, DriftEngineConfig)):
        _model_path = backend_config.model_path if backend_config.model_path is not None else _model_path
        # Convert DriftEngineConfig log_level from lowercase to proper Python logging levels
        if hasattr(backend_config, 'log_level') and backend_config.log_level is not None:
            if isinstance(backend_config, DriftEngineConfig):
                _log_level = backend_config.log_level.upper()
            else:
                _log_level = backend_config.log_level
        else:
            _log_level = _log_level
        _speculative_config = backend_config.speculative_config if hasattr(backend_config, 'speculative_config') and backend_config.speculative_config is not None else _speculative_config
        _backend_config_for_engine = backend_config
    elif isinstance(backend_config, TurbomindEngineConfig):
        # Ensure speculative_config is set if it exists in the top-level
        if backend_config.speculative_config is None:
            backend_config.speculative_config = _speculative_config
        _backend_config_for_engine = backend_config
    elif isinstance(backend_config, PytorchEngineConfig):
        _backend_config_for_engine = backend_config
    else:
        # Fallback to autoget_backend_config if no specific config is provided
        _backend_config_for_engine = autoget_backend_config(_model_path, backend_config)
    
    # Update logger level based on possibly updated _log_level
    logger.setLevel(_log_level)

    # model_path for get_model
    # This ensures that the model_path used for downloading is the most relevant one
    if hasattr(_backend_config_for_engine, 'model_path') and _backend_config_for_engine.model_path is not None:
        _model_path = _backend_config_for_engine.model_path
    
    # model_path is not local path.
    if not os.path.exists(_model_path):
        download_dir = _backend_config_for_engine.download_dir \
            if hasattr(_backend_config_for_engine, 'download_dir') else None
        revision = _backend_config_for_engine.revision \
            if hasattr(_backend_config_for_engine, 'revision') else None
        _model_path = get_model(_model_path, download_dir, revision)
        if hasattr(_backend_config_for_engine, 'model_path'):
            _backend_config_for_engine.model_path = _model_path # Update model_path in the config too

    # spec model
    # Handle speculative config for TurbomindEngineConfig and TurboMindEngineConfig
    if isinstance(_backend_config_for_engine, (TurbomindEngineConfig, TurboMindEngineConfig, DriftEngineConfig)):
        _spec_cfg_to_use = _backend_config_for_engine.speculative_config
        if _spec_cfg_to_use is not None and _spec_cfg_to_use.model and not os.path.exists(_spec_cfg_to_use.model):
            download_dir = _backend_config_for_engine.download_dir \
                if hasattr(_backend_config_for_engine, 'download_dir') else None
            _spec_cfg_to_use.model = get_model(_spec_cfg_to_use.model, download_dir)
            _backend_config_for_engine.speculative_config = _spec_cfg_to_use # Update in config
    elif _speculative_config is not None and _speculative_config.model and not os.path.exists(_speculative_config.model):
        download_dir = _backend_config_for_engine.download_dir \
            if hasattr(_backend_config_for_engine, 'download_dir') else None
        _speculative_config.model = get_model(_speculative_config.model, download_dir)
        
    _, pipeline_class = get_task(_model_path)
    backend = 'pytorch' if isinstance(_backend_config_for_engine, PytorchEngineConfig) else 'drift' if isinstance(_backend_config_for_engine, DriftEngineConfig) else 'turbomind'
    logger.info(f'Using {backend} engine')

    return pipeline_class(_model_path,
                          backend=backend,
                          backend_config=_backend_config_for_engine,
                          chat_template_config=chat_template_config,
                          max_log_len=max_log_len,
                          speculative_config=_speculative_config)


def drift_pipeline(model_path: str,
                   backend_config: Optional[DriftEngineConfig] = None,
                   chat_template_config: Optional[ChatTemplateConfig] = None,
                   log_level: str = 'WARNING',
                   max_log_len: int = None,
                   speculative_config: SpeculativeConfig = None):
    """
    Convenience wrapper for pipeline with DriftEngineConfig.
    """
    backend_config = backend_config or DriftEngineConfig(model_path=model_path)
    return pipeline(model_path,
                    backend_config=backend_config,
                    chat_template_config=chat_template_config,
                    log_level=log_level,
                    max_log_len=max_log_len,
                    speculative_config=speculative_config)


def serve(model_path: str,
          model_name: Optional[str] = None,
          backend: Literal['turbomind', 'pytorch', 'drift'] = 'turbomind',
          backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig, TurboMindEngineConfig, DriftEngineConfig]] = None,
          chat_template_config: Optional[ChatTemplateConfig] = None,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          log_level: str = 'ERROR',
          api_keys: Optional[Union[List[str], str]] = None,
          ssl: bool = False):
    """This will run the api_server in a subprocess.

    Args:
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): the name of the served model. It can be accessed
            by the RESTful API `/v1/models`. If it is not specified,
            `model_path` will be adopted
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig | DriftEngineConfig): backend
            config instance. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        server_name (str): host ip for serving
        server_port (int): server port
        log_level(str): set log level whose value among [CRITICAL, ERROR, WARNING, INFO, DEBUG]
        api_keys (List[str] | str | None): Optional list of API keys. Accepts string type as
            a single api_key. Default to None, which means no api key applied.
        ssl (bool): Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.

    Return:
        APIClient: A client chatbot for LLaMA series models.

    Examples:
        >>> import lmdeploy
        >>> client = lmdeploy.serve('internlm/internlm-chat-7b', 'internlm-chat-7b')
        >>> for output in client.chat('hi', 1):
        ...    print(output)
    """ # noqa E501
    import time
    from multiprocessing import Process

    from lmdeploy.serve.openai.api_client import APIClient
    from lmdeploy.serve.openai.api_server import serve as run_api_server # Rename the imported serve to avoid conflict

    _model_path = model_path
    _log_level = log_level
    _backend_config_for_engine = backend_config

    if isinstance(backend_config, (TurboMindEngineConfig, DriftEngineConfig)):
        _model_path = backend_config.model_path if backend_config.model_path is not None else _model_path
        _log_level = backend_config.log_level if backend_config.log_level is not None else _log_level
        _backend_config_for_engine = backend_config
    elif isinstance(backend_config, TurbomindEngineConfig):
        _backend_config_for_engine = backend_config
    elif isinstance(backend_config, PytorchEngineConfig):
        _backend_config_for_engine = backend_config
    else:
        # Fallback to autoget_backend_config if no specific config is provided
        _backend_config_for_engine = autoget_backend_config(_model_path, backend_config)
    
    # model_path for get_model
    if hasattr(_backend_config_for_engine, 'model_path') and _backend_config_for_engine.model_path is not None:
        _model_path = _backend_config_for_engine.model_path
    
    # model_path is not local path.
    if not os.path.exists(_model_path):
        download_dir = _backend_config_for_engine.download_dir \
            if hasattr(_backend_config_for_engine, 'download_dir') else None
        revision = _backend_config_for_engine.revision \
            if hasattr(_backend_config_for_engine, 'revision') else None
        _model_path = get_model(_model_path, download_dir, revision)
        if hasattr(_backend_config_for_engine, 'model_path'):
            _backend_config_for_engine.model_path = _model_path # Update model_path in the config too
    
    backend_val = 'pytorch' if isinstance(_backend_config_for_engine, PytorchEngineConfig) else 'drift' if isinstance(_backend_config_for_engine, DriftEngineConfig) else 'turbomind'

    task = Process(target=run_api_server, # Use the renamed imported serve
                   args=(_model_path, ),
                   kwargs=dict(model_name=model_name,
                               backend=backend_val,
                               backend_config=_backend_config_for_engine,
                               chat_template_config=chat_template_config,
                               server_name=server_name,
                               server_port=server_port,
                               log_level=_log_level,
                               api_keys=api_keys,
                               ssl=ssl),
                   daemon=True)
    task.start()
    client = APIClient(f'http://{server_name}:{server_port}')
    while True:
        time.sleep(1)
        try:
            client.available_models
            print(f'Launched the api_server in process {task.pid}, user can '
                  f'kill the server by:\nimport os,signal\nos.kill({task.pid}, '
                  'signal.SIGKILL)')
            return client
        except:  # noqa
            pass

def drift_api_server(model_path: str,
                     model_name: Optional[str] = None,
                     backend_config: Optional[DriftEngineConfig] = None,
                     chat_template_config: Optional[ChatTemplateConfig] = None,
                     server_name: str = '0.0.0.0',
                     server_port: int = 23333,
                     log_level: str = 'ERROR',
                     api_keys: Optional[Union[List[str], str]] = None,
                     ssl: bool = False):
    """
    Convenience wrapper for serve with DriftEngineConfig.
    """
    backend_config = backend_config or DriftEngineConfig(model_path=model_path)
    return serve(model_path,
                 model_name=model_name,
                 backend='drift',
                 backend_config=backend_config,
                 chat_template_config=chat_template_config,
                 server_name=server_name,
                 server_port=server_port,
                 log_level=log_level,
                 api_keys=api_keys,
                 ssl=ssl)


def client(api_server_url: str = 'http://0.0.0.0:23333', api_key: Optional[str] = None, **kwargs):
    """
    Args:
        api_server_url (str): communicating address 'http://<ip>:<port>' of
            api_server
        api_key (str | None): api key. Default to None, which means no
            api key will be used.
    Return:
        Chatbot for LLaMA series models with turbomind as inference engine.
    """
    from lmdeploy.serve.openai.api_client import APIClient
    return APIClient(api_server_url, api_key, **kwargs)

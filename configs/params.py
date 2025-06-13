# -*- coding: UTF-8 -*-
from configparser import ConfigParser


class BaseParams(object):
    def __init__(self, conf_fp: str = 'configs/config.ini'):
        self.config = ConfigParser()
        self.config.read(conf_fp, encoding='utf8')


class ModelParams(BaseParams):
    def __init__(self, conf_fp: str = 'configs/config.ini'):
        super(ModelParams, self).__init__(conf_fp)
        section_name = 'local_model_configs'
        self.model_path = self.config.get(section_name, 'model_path')
        self.temperature = self.config.getfloat(section_name, 'temperature')
        self.max_tokens = self.config.getint(section_name, 'max_tokens')
        

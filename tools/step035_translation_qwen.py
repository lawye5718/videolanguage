# -*- coding: utf-8 -*-
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

extra_body = {
    'repetition_penalty': 1.1,
}

def get_llm_api_config():
    """
    通用的大模型API配置加载函数
    优先读取 cinecast 项目中的LLM配置，支持多种模型提供商
    """
    # 假设 videolanguage 和 cinecast 在同一个父目录下
    # 例如：
    # /workspace/cinecast/
    # /workspace/videolanguage/
    
    # 1. 首先检查新的LLM配置文件（WebUI可能修改的）
    cinecast_llm_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../cinecast/.cinecast_llm_config.json")
    )
    if os.path.exists(cinecast_llm_config_path):
        try:
            with open(cinecast_llm_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"✅ 从 cinecast LLM 配置文件加载: {cinecast_llm_config_path}")
                logger.info(f"🔍 配置内容: {config}")
                return (
                    config.get("api_key", ""),
                    config.get("base_url", "https://api.openai.com/v1"),
                    config.get("model_name", "gpt-3.5-turbo")
                )
        except Exception as e:
            logger.warning(f"⚠️ 读取 cinecast LLM 配置文件失败: {e}")
    
    # 2. 回退到加载cinecast的.env文件
    cinecast_env_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../cinecast/.env")
    )
    if os.path.exists(cinecast_env_path):
        load_dotenv(cinecast_env_path)
        logger.info(f"✅ 已加载 cinecast .env 文件: {cinecast_env_path}")
    
    # 3. 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
    if api_key:
        logger.info(f"✅ 从环境变量获取 API Key: {api_key[:10]}...")
        # 根据API密钥前缀判断提供商
        if api_key.startswith("sk-5bc8c199"):
            # DeepSeek
            return api_key, "https://api.deepseek.com/v1", "deepseek-chat"
        else:
            # 默认Qwen
            return api_key, "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.5-plus"
    
    # 4. 最后回退到旧的配置文件
    cinecast_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../cinecast/qwen_api_config.json")
    )
    
    logger.info(f"⚠️ 未找到环境变量，尝试读取旧配置文件")
    logger.info(f"🔍 正在读取配置文件: {cinecast_config_path}")
    logger.info(f"🔍 文件是否存在: {os.path.exists(cinecast_config_path)}")
    
    api_key = None
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name = "qwen3.5-plus"

    if os.path.exists(cinecast_config_path):
        try:
            with open(cinecast_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"🔍 旧配置文件内容: {config}")
                api_key = config.get("api_key", config.get("QWEN_API_KEY", ""))
                model_name = config.get("model", model_name)
                if "base_url" in config:
                    base_url = config["base_url"]
            logger.info(f"✅ 成功加载旧 Cinecast 配置文件: {cinecast_config_path}")
        except Exception as e:
            logger.warning(f"⚠️ 读取旧 Cinecast 配置文件失败: {e}")
            
    # 如果都没读到，尝试从系统环境变量获取
    if not api_key:
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        
    if not api_key:
        raise ValueError("❌ 无法找到 API Key，请检查 cinecast 配置或设置环境变量")
        
    return api_key, base_url, model_name

def llm_response(messages):
    api_key, base_url, model_name = get_llm_api_config()
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        timeout=240,
        extra_body=extra_body
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    response = llm_response(test_message)
    print(response)
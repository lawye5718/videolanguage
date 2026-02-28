# -*- coding: utf-8 -*-
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

extra_body = {
    'repetition_penalty': 1.1,
}

def get_qwen_api_config():
    """
    优先读取 cinecast 项目中的环境变量 DASHSCOPE_API_KEY，
    如果没有，则回退到读取 qwen_api_config.json 配置文件。
    """
    # 假设 videolanguage 和 cinecast 在同一个父目录下
    # 例如：
    # /workspace/cinecast/
    # /workspace/videolanguage/
    
    # 显式加载cinecast的.env文件（WebUI修改的配置在这里）
    cinecast_env_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../cinecast/.env")
    )
    if os.path.exists(cinecast_env_path):
        load_dotenv(cinecast_env_path)
        logger.info(f"✅ 已加载 cinecast .env 文件: {cinecast_env_path}")
    
    # 优先检查环境变量（与cinecast WebUI保持同步）
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        logger.info(f"✅ 从环境变量 DASHSCOPE_API_KEY 获取 API Key: {api_key[:10]}...")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model_name = "qwen3.5-plus"
        return api_key, base_url, model_name
    
    # 回退到配置文件
    cinecast_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../cinecast/qwen_api_config.json")
    )
    
    logger.info(f"⚠️ 未找到环境变量 DASHSCOPE_API_KEY，尝试读取配置文件")
    logger.info(f"🔍 正在读取配置文件: {cinecast_config_path}")
    logger.info(f"🔍 文件是否存在: {os.path.exists(cinecast_config_path)}")
    
    api_key = None
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 默认通义千问兼容 OpenAI 的地址
    model_name = "qwen3.5-plus" # 默认模型

    if os.path.exists(cinecast_config_path):
        try:
            with open(cinecast_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"🔍 配置文件内容: {config}")
                # 根据 cinecast json 的实际 key 名称来获取
                api_key = config.get("api_key", config.get("QWEN_API_KEY", ""))
                model_name = config.get("model", model_name)
                if "base_url" in config:
                    base_url = config["base_url"]
            logger.info(f"✅ [Qwen Translation] 成功加载 Cinecast 配置文件: {cinecast_config_path}")
        except Exception as e:
            logger.warning(f"⚠️ 读取 Cinecast 配置文件失败: {e}")
            
    # 如果配置文件没读到，尝试从系统环境变量获取
    if not api_key:
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        
    if not api_key:
        raise ValueError("❌ 无法找到 Qwen API Key，请检查 cinecast 配置或设置环境变量 QWEN_API_KEY")
        
    return api_key, base_url, model_name

def qwen_response(messages):
    api_key, base_url, model_name = get_qwen_api_config()
    
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
    response = qwen_response(test_message)
    print(response)
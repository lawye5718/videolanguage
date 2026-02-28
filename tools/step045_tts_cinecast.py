import os
import requests
from loguru import logger

def generate_tts_cinecast(text, output_path, voice_id="aiden"):
    """
    调用本地 Mac mini 上的 Cinecast 兼容 OpenAI 格式 API
    """
    url = "http://localhost:8888/v1/audio/speech"
    payload = {
        "model": "qwen3-tts",
        "input": text,
        "voice": voice_id,
        "response_format": "mp3"
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        # 流式写入文件
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"[Cinecast TTS] 成功生成音频: {output_path}")
        return True
    except Exception as e:
        logger.error(f"[Cinecast TTS] API 调用失败: {e}")
        return False

# 用于测试
if __name__ == "__main__":
    test_text = "这是Cinecast TTS集成测试"
    output_file = "test_cinecast_tts.mp3"
    success = generate_tts_cinecast(test_text, output_file)
    if success:
        print(f"测试成功，音频文件已保存为: {output_file}")
    else:
        print("测试失败")
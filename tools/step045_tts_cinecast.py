import os
import requests
from loguru import logger
from pydub import AudioSegment

# 您的 cinecast 本地 API 地址
CINECAST_API_URL = "http://localhost:8888"

def get_padded_reference_audio(audio_segment, start_sec, end_sec, min_duration=4.0):
    """
    智能切片：提取带有情绪的参考音频。
    如果片段太短（<4秒），则向前后扩展上下文，以保证 Qwen3-TTS 提取到稳定的情绪特征。
    """
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    duration_ms = end_ms - start_ms
    min_duration_ms = int(min_duration * 1000)

    if duration_ms < min_duration_ms:
        # 计算需要补充的毫秒数，平均分摊到前后
        pad_ms = (min_duration_ms - duration_ms) // 2
        start_ms = max(0, start_ms - pad_ms)
        end_ms = min(len(audio_segment), end_ms + pad_ms)
        
    return audio_segment[start_ms:end_ms]

def generate_tts_with_emotion_clone(text, start_time, end_time, vocal_audio_path, output_audio_path, emotion_voice="aiden"):
    """
    调用 Cinecast API 实现逐句带情绪的配音（使用参考音频进行音色克隆）
    """
    logger.info(f"🎤 [情绪配音] 准备生成: {text[:15]}...")
    
    try:
        full_vocal = AudioSegment.from_wav(vocal_audio_path)
        ref_segment = get_padded_reference_audio(full_vocal, start_time, end_time)
        temp_ref_path = output_audio_path.replace(".mp3", "_ref.wav")
        ref_segment.export(temp_ref_path, format="wav")
    except Exception as e:
        logger.error(f"❌ [情绪配音] 提取参考音频失败: {e}")
        return False

    url = f"{CINECAST_API_URL}/v1/audio/speech"
    
    try:
        # 暂时使用简单的JSON请求（不含参考音频）
        payload = {
            "model": "qwen3-tts",
            "input": text,
            "voice": emotion_voice,
            "response_format": "mp3"
        }
        
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        with open(output_audio_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: 
                    f.write(chunk)
                    
        logger.info(f"✅ [情绪配音] 成功生成配音: {output_audio_path} (音色: {emotion_voice})")
        return True
    
    except Exception as e:
        logger.error(f"❌ [情绪配音] API 调用或处理失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)

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
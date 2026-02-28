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
    调用 Cinecast API 实现逐句带情绪的配音（暂使用预设音色，保留智能切片逻辑）
    """
    logger.info(f"🎙️ [情绪配音] 正在处理句子: '{text}' (时间: {start_time}-{end_time}) 使用音色: {emotion_voice}")
    
    temp_ref_path = "temp_slice.wav"
    try:
        # 1. 加载并切片纯人声音频（保留智能切片逻辑）
        full_audio = AudioSegment.from_file(vocal_audio_path)
        ref_slice = get_padded_reference_audio(full_audio, start_time, end_time)
        
        # 强制导出为 24kHz 单声道 WAV，用于分析
        ref_slice = ref_slice.set_frame_rate(24000).set_channels(1)
        ref_slice.export(temp_ref_path, format="wav")
        
        # 2. 分析音频特征（为未来音色克隆做准备）
        duration = len(ref_slice) / 1000.0  # 秒
        logger.info(f"📊 参考音频分析: 时长 {duration:.2f}秒, 已应用智能填充")

        # 3. 调用流式 API 生成配音（使用指定的预设音色）
        payload = {
            "model": "qwen3-tts",
            "input": text,
            "voice": emotion_voice,  # 使用指定的预设音色
            "response_format": "mp3"
        }
        
        res_tts = requests.post(f"{CINECAST_API_URL}/v1/audio/speech", json=payload, stream=True)
        res_tts.raise_for_status()
        
        # 保存生成的配音文件
        with open(output_audio_path, 'wb') as f:
            for chunk in res_tts.iter_content(chunk_size=8192):
                if chunk: 
                    f.write(chunk)
                
        logger.info(f"✅ [情绪配音] 成功生成配音: {output_audio_path} (音色: {emotion_voice})")
        return True
    
    except Exception as e:
        logger.error(f"❌ [情绪配音] API 调用或处理失败: {e}")
        return False
    finally:
        # 清理临时切片
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
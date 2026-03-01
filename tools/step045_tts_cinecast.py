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
    
    # 提取参考音频
    temp_ref_path = output_audio_path.replace(".wav", "_ref.wav")
    try:
        full_vocal = AudioSegment.from_wav(vocal_audio_path)
        ref_segment = get_padded_reference_audio(full_vocal, start_time, end_time)
        ref_segment.export(temp_ref_path, format="wav")
    except Exception as e:
        logger.error(f"❌ [情绪配音] 提取参考音频失败: {e}")
        return False

    url = f"{CINECAST_API_URL}/v1/audio/speech"
    
    # 构造兼容的 Form 数据
    data = {
        "model": "qwen3-tts",
        "input": str(text),
        "voice": str(emotion_voice),
        "response_format": "mp3"
    }
    
    try:
        with open(temp_ref_path, 'rb') as ref_file:
            files = {
                'reference_audio': ('ref.wav', ref_file, 'audio/wav')
            }
            
            # 使用 data 和 files，触发带有参考音频的情感克隆
            response = requests.post(url, data=data, files=files, stream=True)
            
            if response.status_code != 200:
                logger.error(f"❌ 详细的API拒绝原因: {response.text}")
            response.raise_for_status()
            
            # 💡 【关键修复】：先保存为 mp3
            temp_mp3_path = output_audio_path.replace(".wav", ".mp3")
            with open(temp_mp3_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: 
                        f.write(chunk)
            
            # 💡 【关键修复】：将其转换为血统纯正的 WAV 格式，供 librosa 读取
            AudioSegment.from_file(temp_mp3_path).export(output_audio_path, format="wav")
            
            # 清理临时的 mp3 文件
            if os.path.exists(temp_mp3_path):
                os.remove(temp_mp3_path)
                    
        logger.info(f"✅ [情绪配音] 成功生成配音: {output_audio_path} (音色: {emotion_voice})")
        return True
    
    except Exception as e:
        logger.error(f"❌ [情绪配音] API 调用或处理失败: {e}")
        return False
    finally:
        # 清理临时的参考音频文件
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)

def generate_tts_cinecast(text, output_path, voice_id="aiden"):
    """
    备用：普通文本转语音调用
    """
    url = f"{CINECAST_API_URL}/v1/audio/speech"
    data = {
        "model": "qwen3-tts",
        "input": str(text),
        "voice": str(voice_id),
        "response_format": "mp3"
    }
    
    try:
        files = {'dummy': ('', '')}
        response = requests.post(url, data=data, files=files, stream=True)
        if response.status_code != 200:
            logger.error(f"❌ 详细的API拒绝原因: {response.text}")
        response.raise_for_status()
        
        # 同步应用格式转换修复
        temp_mp3_path = output_path.replace(".wav", ".mp3")
        with open(temp_mp3_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
                
        AudioSegment.from_file(temp_mp3_path).export(output_path, format="wav")
        if os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)
            
        logger.info(f"[Cinecast TTS] 成功生成音频: {output_path}")
        return True
    except Exception as e:
        logger.error(f"[Cinecast TTS] API 调用失败: {e}")
        return False
import json
import time
import librosa
import numpy as np
import whisperx
import os
from loguru import logger
import torch
from dotenv import load_dotenv
load_dotenv()

whisper_model = None
diarize_model = None

align_model = None
language_code = None
align_metadata = None

def init_whisperx():
    load_whisper_model()
    load_align_model()

def init_diarize():
    load_diarize_model()
    
def load_whisper_model(model_name: str = 'large', download_root = 'models/ASR/whisper', device='auto'):
    if model_name == 'large':
        pretrain_model = os.path.join(download_root,"faster-whisper-large-v3")
        model_name = 'large-v3' if not os.path.isdir(pretrain_model) else pretrain_model
        
    global whisper_model
    if whisper_model is not None:
        return
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
            compute_type = 'float32'  # Mac MPS 对 float16 支持不完善
        else:
            device = 'cpu'
            compute_type = 'int8'
    logger.info(f'Loading WhisperX model: {model_name}')
    t_start = time.time()
    if device == 'cpu':
        whisper_model = whisperx.load_model(model_name, download_root=download_root, device=device, compute_type=compute_type)
    elif device == 'mps':
        whisper_model = whisperx.load_model(model_name, download_root=download_root, device=device, compute_type=compute_type)
    else:
        whisper_model = whisperx.load_model(model_name, download_root=download_root, device=device)
    t_end = time.time()
    logger.info(f'Loaded WhisperX model: {model_name} in {t_end - t_start:.2f}s')

def load_align_model(language='en', device='auto', model_dir='models/ASR/whisper'):
    global align_model, language_code, align_metadata
    if align_model is not None and language_code == language:
        return
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    language_code = language
    t_start = time.time()
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language_code, device=device, model_dir = model_dir)
    t_end = time.time()
    logger.info(f'Loaded alignment model: {language_code} in {t_end - t_start:.2f}s')
    
def load_diarize_model(device='auto'):
    global diarize_model
    if diarize_model is not None:
        return
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # 检查HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.warning("⚠️ 未在 .env 中检测到 HF_TOKEN，跳过说话人分离。如果需要多角色配音，请配置 HF_TOKEN。")
        return
    
    t_start = time.time()
    try:
        # 注意：Mac MPS 在 Pyannote 某些算子上可能报错，如果这里崩溃，请将 device 改为 "cpu"
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        t_end = time.time()
        logger.info(f'✅ Loaded diarization model in {t_end - t_start:.2f}s')
    except Exception as e:
        t_end = time.time()
        logger.warning(f"⚠️ 说话人分离失败 (可能因为 Mac 算子不兼容): {str(e)}")
        logger.info("👉 将回退到不区分说话人的单角色模式。")
        logger.info("💡 建议：如果需要说话人分离功能，可在 .env 中设置 HF_TOKEN 并将 device 改为 cpu")

def whisperx_transcribe_audio(wav_path, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True,min_speakers=None, max_speakers=None):
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    logger.info(f"▶️ 开始 WhisperX 语音识别 (设备: {device})...")
    load_whisper_model(model_name, download_root, device)
    rec_result = whisper_model.transcribe(wav_path, batch_size=batch_size)
    
    if rec_result['language'] == 'nn':
        logger.warning(f'No language detected in {wav_path}')
        return False
    
    logger.info("▶️ 开始时间戳对齐...")
    load_align_model(rec_result['language'])
    rec_result = whisperx.align(rec_result['segments'], align_model, align_metadata,
                                wav_path, device, return_char_alignments=False)
    
    if diarization:
        logger.info("▶️ 开始说话人分离 (Diarization)...")
        load_diarize_model(device)
        if diarize_model:
            try:
                diarize_segments = diarize_model(wav_path,min_speakers=min_speakers, max_speakers=max_speakers)
                rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)
                logger.info("✅ 说话人分离完成")
            except Exception as e:
                logger.warning(f"⚠️ 说话人分离执行失败: {str(e)}")
                logger.info("👉 继续使用无说话人标记的结果")
        else:
            logger.info("ℹ️ 未启用说话人分离，使用单角色模式")
        
    transcript = [{'start': segement['start'], 'end': segement['end'], 'text': segement['text'].strip(), 'speaker': segement.get('speaker', 'SPEAKER_00')} for segement in rec_result['segments']]
    return transcript


if __name__ == '__main__':
    for root, dirs, files in os.walk("videos"):
        if 'audio_vocals.wav' in files:
            logger.info(f'Transcribing {os.path.join(root, "audio_vocals.wav")}')
            transcript = whisperx_transcribe_audio(os.path.join(root, "audio_vocals.wav"))
            print(transcript)
            break
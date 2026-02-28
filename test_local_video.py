#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoLanguage 本地视频测试脚本
测试视频: /Users/yuanliang/Downloads/p.mp4
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志配置"""
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add("test_log.log", rotation="100 MB", level="DEBUG")

def check_dependencies():
    """检查依赖环境"""
    logger.info("🔍 检查依赖环境...")
    
    # 检查FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ FFmpeg 可用")
        else:
            logger.error("❌ FFmpeg 不可用")
            return False
    except FileNotFoundError:
        logger.error("❌ FFmpeg 未安装")
        return False
    
    # 检查Cinecast API
    try:
        import requests
        response = requests.get("http://localhost:8888/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Cinecast API 服务运行中")
        else:
            logger.error("❌ Cinecast API 服务异常")
            return False
    except Exception as e:
        logger.error(f"❌ Cinecast API 连接失败: {e}")
        return False
    
    return True

def prepare_test_video():
    """准备测试视频"""
    video_path = "/Users/yuanliang/Downloads/p.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"❌ 测试视频不存在: {video_path}")
        return None
    
    # 创建测试工作目录
    test_dir = project_root / "test_output"
    test_dir.mkdir(exist_ok=True)
    
    # 复制视频到工作目录
    import shutil
    test_video = test_dir / "test_video.mp4"
    if not test_video.exists():
        shutil.copy2(video_path, test_video)
        logger.info(f"✅ 测试视频已复制到: {test_video}")
    
    return str(test_video)

def run_step010_demucs(video_path):
    """运行人声分离"""
    logger.info("🎵 [Step 010] 开始人声分离...")
    
    try:
        from tools.step010_demucs_vr import separate_all_audio_under_folder
        
        # 创建处理目录结构
        video_dir = Path(video_path).parent
        audio_dir = video_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # 复制视频到音频目录
        import shutil
        video_copy = audio_dir / "video.mp4"
        if not video_copy.exists():
            shutil.copy2(video_path, video_copy)
        
        # 执行分离
        result, vocal_path, instrumental_path = separate_all_audio_under_folder(str(video_dir))
        logger.info(f"✅ 人声分离完成: {result}")
        logger.info(f"   人声轨道: {vocal_path}")
        logger.info(f"   背景音轨道: {instrumental_path}")
        
        return vocal_path, instrumental_path
        
    except Exception as e:
        logger.error(f"❌ 人声分离失败: {e}")
        return None, None

def run_step021_asr(vocal_path):
    """运行语音识别"""
    logger.info("🎤 [Step 021] 开始语音识别...")
    
    try:
        from tools.step021_asr_whisperx import whisperx_transcribe_audio
        
        if not vocal_path or not os.path.exists(vocal_path):
            logger.error("❌ 人声轨道文件不存在")
            return None
            
        # 执行语音识别
        transcript = whisperx_transcribe_audio(
            wav_path=vocal_path,
            model_name='large',
            device='mps',  # 使用MPS加速
            diarization=True  # 启用说话人分离
        )
        
        if transcript:
            logger.info(f"✅ 语音识别完成，识别到 {len(transcript)} 条字幕")
            
            # 保存字幕文件
            import json
            output_dir = Path(vocal_path).parent.parent
            subtitle_file = output_dir / "subtitles.json"
            with open(subtitle_file, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
            logger.info(f"📄 字幕已保存到: {subtitle_file}")
            
            return transcript
        else:
            logger.error("❌ 语音识别返回空结果")
            return None
            
    except Exception as e:
        logger.error(f"❌ 语音识别失败: {e}")
        return None

def run_step035_translation(transcript, target_language="zh"):
    """运行翻译"""
    logger.info(f"🔤 [Step 035] 开始翻译为 {target_language}...")
    
    try:
        from tools.step035_translation_qwen import qwen_response
        
        if not transcript:
            logger.error("❌ 字幕数据为空")
            return None
            
        # 提取原文本
        texts = [item['text'] for item in transcript[:10]]  # 取前10句测试
        text_batch = "\n".join(texts)
        
        # 构造翻译提示
        prompt = f"""你是一个专业的视频字幕翻译员。请将以下文本翻译为{target_language}，保持语境连贯，不要添加任何额外的解释。

{text_batch}"""
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 调用Qwen翻译
        translation = qwen_response(messages)
        
        if translation:
            logger.info("✅ 翻译完成")
            logger.info(f"翻译结果预览: {translation[:100]}...")
            
            # 保存翻译结果
            output_dir = Path(transcript[0]['text']).parent.parent if isinstance(transcript[0]['text'], str) else Path.cwd()
            translation_file = output_dir / "translation_result.txt"
            with open(translation_file, 'w', encoding='utf-8') as f:
                f.write(translation)
            logger.info(f"📄 翻译结果已保存到: {translation_file}")
            
            return translation
        else:
            logger.error("❌ 翻译返回空结果")
            return None
            
    except Exception as e:
        logger.error(f"❌ 翻译失败: {e}")
        return None

def run_step045_tts(transcript, vocal_path):
    """运行情绪配音"""
    logger.info("🎙️ [Step 045] 开始情绪配音...")
    
    try:
        from tools.step045_tts_cinecast import generate_tts_with_emotion_clone
        
        if not transcript or not vocal_path:
            logger.error("❌ 缺少必要数据")
            return False
            
        output_dir = Path(vocal_path).parent.parent / "dubbing_output"
        output_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = min(len(transcript), 5)  # 测试前5句
        
        for i, item in enumerate(transcript[:total_count]):
            text = item.get('text', '').strip()
            start_time = item.get('start', 0)
            end_time = item.get('end', start_time + 3)
            
            if not text:
                continue
                
            output_file = output_dir / f"dub_{i:04d}.mp3"
            
            success = generate_tts_with_emotion_clone(
                text=text,
                start_time=start_time,
                end_time=end_time,
                vocal_audio_path=vocal_path,
                output_audio_path=str(output_file),
                emotion_voice="aiden"
            )
            
            if success:
                success_count += 1
                logger.info(f"✅ 第{i+1}句配音完成")
            else:
                logger.error(f"❌ 第{i+1}句配音失败")
        
        logger.info(f"🏁 情绪配音完成: {success_count}/{total_count} 句成功")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ 情绪配音失败: {e}")
        return False

def main():
    """主测试流程"""
    setup_logging()
    logger.info("🚀 开始 VideoLanguage 本地视频测试")
    logger.info(f"测试视频: /Users/yuanliang/Downloads/p.mp4")
    
    # 1. 环境检查
    if not check_dependencies():
        logger.error("❌ 环境依赖检查失败，测试终止")
        return
    
    # 2. 准备测试视频
    video_path = prepare_test_video()
    if not video_path:
        logger.error("❌ 测试视频准备失败")
        return
    
    logger.info(f"✅ 使用测试视频: {video_path}")
    
    # 3. 人声分离
    vocal_path, instrumental_path = run_step010_demucs(video_path)
    if not vocal_path:
        logger.error("❌ 人声分离失败，测试终止")
        return
    
    # 4. 语音识别
    transcript = run_step021_asr(vocal_path)
    if not transcript:
        logger.error("❌ 语音识别失败，测试终止")
        return
    
    # 5. 翻译测试
    translation = run_step035_translation(transcript, "zh")
    if not translation:
        logger.warning("⚠️ 翻译失败，继续后续步骤")
    
    # 6. 情绪配音测试
    tts_success = run_step045_tts(transcript, vocal_path)
    
    # 7. 总结
    logger.info("🎉 测试完成!")
    logger.info("📋 测试结果:")
    logger.info(f"   - 人声分离: {'✅ 成功' if vocal_path else '❌ 失败'}")
    logger.info(f"   - 语音识别: {'✅ 成功' if transcript else '❌ 失败'}")
    logger.info(f"   - 翻译功能: {'✅ 成功' if translation else '❌ 失败'}")
    logger.info(f"   - 情绪配音: {'✅ 成功' if tts_success else '❌ 失败'}")
    
    # 显示输出目录
    output_dirs = [
        Path(video_path).parent,
        Path(vocal_path).parent.parent if vocal_path else None
    ]
    
    for output_dir in output_dirs:
        if output_dir and output_dir.exists():
            logger.info(f"📁 输出目录: {output_dir}")
            files = list(output_dir.glob("*"))
            for file in files[:10]:  # 显示前10个文件
                logger.info(f"      {file.name}")

if __name__ == "__main__":
    main()
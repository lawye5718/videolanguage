#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoLanguage 首秀测试脚本 (First Run Test)
一站式全链路视频处理测试
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add("first_run_test.log", rotation="100 MB", level="DEBUG")

def prepare_test_video():
    """准备测试视频"""
    logger.info("🎬 准备测试视频...")
    
    source_video = "/Users/yuanliang/Downloads/p.mp4"
    if not os.path.exists(source_video):
        logger.error(f"❌ 测试视频不存在: {source_video}")
        return None
    
    # 创建测试工作目录
    test_dir = project_root / "first_run_test"
    test_dir.mkdir(exist_ok=True)
    
    # 复制视频
    test_video = test_dir / "input_video.mp4"
    if not test_video.exists():
        logger.info("🔄 复制测试视频...")
        shutil.copy2(source_video, test_video)
        logger.info(f"✅ 视频已复制到: {test_video}")
    
    return str(test_video)

def run_demucs_separation(video_path):
    """运行人声分离"""
    logger.info("🎵 [步骤1] 运行人声分离 (Demucs)...")
    
    try:
        from tools.step010_demucs_simple import separate_all_audio_under_folder
        
        video_dir = Path(video_path).parent
        audio_dir = video_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # 复制视频文件到audio目录
        import shutil
        video_copy = audio_dir / "video.mp4"
        if not video_copy.exists():
            shutil.copy2(video_path, video_copy)
        
        logger.info("🔄 执行人声分离...")
        start_time = time.time()
        result, vocal_path, instrumental_path = separate_all_audio_under_folder(str(video_dir))
        end_time = time.time()
        
        if result and vocal_path:
            logger.success(f"✅ 人声分离完成! 耗时: {end_time - start_time:.2f}秒")
            logger.info(f"   人声轨道: {vocal_path}")
            vocal_size = os.path.getsize(vocal_path)
            logger.info(f"   文件大小: {vocal_size / (1024*1024):.2f} MB")
            return vocal_path
        else:
            logger.error("❌ 人声分离失败")
            return None
            
    except Exception as e:
        logger.error(f"❌ 人声分离出错: {e}")
        logger.exception("详细错误信息:")
        return None

def run_speech_recognition(vocal_path):
    """运行语音识别"""
    logger.info("🎤 [步骤2] 运行语音识别 (WhisperX)...")
    
    try:
        from tools.step021_asr_whisperx import whisperx_transcribe_audio
        
        logger.info("🔄 执行语音识别...")
        start_time = time.time()
        transcript = whisperx_transcribe_audio(
            wav_path=vocal_path,
            model_name='large',
            device='mps',  # 使用MPS加速
            diarization=True
        )
        end_time = time.time()
        
        if transcript:
            logger.success(f"✅ 语音识别完成! 耗时: {end_time - start_time:.2f}秒")
            logger.info(f"   识别到 {len(transcript)} 条字幕")
            
            # 显示前几条字幕
            logger.info("📝 字幕预览:")
            for i, item in enumerate(transcript[:3]):
                text = item.get('text', '')[:50] + ('...' if len(item.get('text', '')) > 50 else '')
                logger.info(f"   [{i+1}] {item.get('start', 0):.2f}-{item.get('end', 0):.2f}s: {text}")
                if 'speaker' in item:
                    logger.info(f"       说话人: {item['speaker']}")
            
            # 保存字幕
            output_dir = Path(vocal_path).parent.parent
            subtitle_file = output_dir / "transcript.json"
            with open(subtitle_file, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
            logger.info(f"📄 字幕已保存到: {subtitle_file}")
            
            return transcript
        else:
            logger.error("❌ 语音识别返回空结果")
            return None
            
    except Exception as e:
        logger.error(f"❌ 语音识别出错: {e}")
        logger.exception("详细错误信息:")
        return None

def run_translation(transcript):
    """运行翻译"""
    logger.info("🔤 [步骤3] 运行AI翻译...")
    
    try:
        from tools.step035_translation_qwen import get_llm_api_config, llm_response
        
        # 检查LLM配置
        api_key, base_url, model_name = get_llm_api_config()
        logger.info(f"✅ 当前LLM配置:")
        logger.info(f"   模型: {model_name}")
        logger.info(f"   Base URL: {base_url}")
        
        if not transcript:
            logger.error("❌ 字幕数据为空")
            return None
        
        # 取前3句进行翻译演示
        texts = [item['text'] for item in transcript[:3] if item.get('text', '').strip()]
        if not texts:
            logger.error("❌ 没有可翻译的文本")
            return None
        
        text_batch = "\n".join(texts)
        logger.info(f"🔄 翻译 {len(texts)} 句文本...")
        
        prompt = f"""你是一个专业的视频字幕翻译员。请将以下英文文本翻译为简体中文，保持语境连贯：

{text_batch}"""
        
        messages = [{"role": "user", "content": prompt}]
        
        start_time = time.time()
        translation = llm_response(messages)
        end_time = time.time()
        
        if translation:
            logger.success(f"✅ 翻译完成! 耗时: {end_time - start_time:.2f}秒")
            logger.info(f"📝 翻译结果预览: {translation[:100]}...")
            
            # 保存翻译结果
            output_dir = Path(transcript[0]['text']).parent.parent if isinstance(transcript[0]['text'], str) else Path.cwd()
            translation_file = output_dir / "translation_result.txt"
            with open(translation_file, 'w', encoding='utf-8') as f:
                f.write(f"原文:\n{text_batch}\n\n译文:\n{translation}")
            logger.info(f"📄 翻译结果已保存到: {translation_file}")
            
            return translation
        else:
            logger.error("❌ 翻译返回空结果")
            return None
            
    except Exception as e:
        logger.error(f"❌ 翻译出错: {e}")
        logger.exception("详细错误信息:")
        return None

def run_emotion_tts(transcript, vocal_path):
    """运行情绪配音"""
    logger.info("🎙️ [步骤4] 运行情绪配音 (Cinecast)...")
    
    try:
        from tools.step045_tts_cinecast import generate_tts_with_emotion_clone
        
        if not transcript or not vocal_path:
            logger.error("❌ 缺少必要数据")
            return False
        
        output_dir = Path(vocal_path).parent.parent / "dubbing_output"
        output_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = min(len(transcript), 2)  # 测试前2句
        
        logger.info(f"🔄 生成 {total_count} 句情绪配音...")
        
        for i, item in enumerate(transcript[:total_count]):
            text = item.get('text', '').strip()
            start_time = item.get('start', 0)
            end_time = item.get('end', start_time + 3)
            
            if not text:
                continue
            
            output_file = output_dir / f"dub_{i:04d}.mp3"
            
            logger.info(f"🔊 生成第{i+1}句情绪配音: '{text[:30]}...'")
            
            success = generate_tts_with_emotion_clone(
                text=text,
                start_time=start_time,
                end_time=end_time,
                vocal_audio_path=vocal_path,
                output_audio_path=str(output_file),
                emotion_voice="aiden"
            )
            
            if success and output_file.exists():
                file_size = output_file.stat().st_size
                logger.success(f"✅ 第{i+1}句配音完成 ({file_size} 字节)")
                success_count += 1
            else:
                logger.error(f"❌ 第{i+1}句配音失败")
        
        logger.info(f"🏁 情绪配音完成: {success_count}/{total_count} 句成功")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ 情绪配音出错: {e}")
        logger.exception("详细错误信息:")
        return False

def generate_final_report(results):
    """生成最终报告"""
    logger.info("📋 生成首秀测试报告...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_video": "/Users/yuanliang/Downloads/p.mp4",
        "steps": {
            "demucs_separation": results[0],
            "speech_recognition": results[1],
            "translation": results[2],
            "emotion_tts": results[3]
        },
        "summary": {}
    }
    
    # 计算成功率
    total_steps = len(results)
    passed_steps = sum(1 for result in results if result)
    success_rate = (passed_steps / total_steps) * 100 if total_steps > 0 else 0
    
    report["summary"] = {
        "total_steps": total_steps,
        "successful_steps": passed_steps,
        "failed_steps": total_steps - passed_steps,
        "success_rate": f"{success_rate:.1f}%"
    }
    
    # 保存报告
    report_file = project_root / "first_run_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.success("🎉 首秀测试完成!")
    logger.info("📊 测试结果汇总:")
    steps_names = ["人声分离", "语音识别", "AI翻译", "情绪配音"]
    for i, (step_result, step_name) in enumerate(zip(results, steps_names)):
        status = "✅ 成功" if step_result else "❌ 失败"
        logger.info(f"   {step_name}: {status}")
    
    logger.info(f"📈 总体成功率: {passed_steps}/{total_steps} ({success_rate:.1f}%)")
    logger.info(f"📄 详细报告已保存到: {report_file}")
    
    return report

def main():
    """主测试流程"""
    setup_logging()
    logger.success("🚀 VideoLanguage 首秀测试开始!")
    logger.info("测试视频: /Users/yuanliang/Downloads/p.mp4")
    logger.info("环境: videolang (conda-forge, Python 3.10.19, MPS支持)")
    
    results = [False, False, False, False]  # [demucs, whisperx, translation, tts]
    
    # 1. 准备测试视频
    video_path = prepare_test_video()
    if not video_path:
        logger.error("❌ 视频准备失败")
        generate_final_report(results)
        return
    
    # 2. 人声分离
    vocal_path = run_demucs_separation(video_path)
    results[0] = bool(vocal_path)
    
    # 3. 语音识别
    if vocal_path:
        transcript = run_speech_recognition(vocal_path)
        results[1] = bool(transcript)
    else:
        logger.warning("⏭️  跳过语音识别（缺少人声轨道）")
    
    # 4. 翻译
    if results[1]:  # 如果语音识别成功
        translation = run_translation(transcript)
        results[2] = bool(translation)
    else:
        logger.warning("⏭️  跳过翻译（缺少字幕数据）")
    
    # 5. 情绪配音
    if results[1]:  # 如果语音识别成功
        tts_success = run_emotion_tts(transcript, vocal_path)
        results[3] = tts_success
    else:
        logger.warning("⏭️  跳过情绪配音（缺少必要数据）")
    
    # 6. 生成最终报告
    generate_final_report(results)

if __name__ == "__main__":
    main()
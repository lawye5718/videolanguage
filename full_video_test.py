#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoLanguage 完整视频处理测试脚本
测试视频: /Users/yuanliang/Downloads/p.mp4
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
    """设置详细的日志配置"""
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add("video_test_detailed.log", rotation="100 MB", level="DEBUG")
    logger.info("🎬 开始 VideoLanguage 完整视频处理测试")

def check_environment():
    """检查运行环境"""
    logger.info("🔍 检查运行环境...")
    
    checks = []
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"✅ Python版本: {python_version}")
    checks.append(True)
    
    # 检查必需依赖
    required_packages = ['requests', 'numpy', 'loguru', 'librosa']
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} 可用")
            checks.append(True)
        except ImportError:
            logger.error(f"❌ {package} 未安装")
            checks.append(False)
    
    # 检查系统工具
    import subprocess
    tools = ['ffmpeg', 'ffprobe']
    for tool in tools:
        try:
            result = subprocess.run([tool, '-version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"✅ {tool} 可用")
                checks.append(True)
            else:
                logger.error(f"❌ {tool} 不可用")
                checks.append(False)
        except Exception as e:
            logger.error(f"❌ {tool} 检查失败: {e}")
            checks.append(False)
    
    return all(checks)

def prepare_test_video():
    """准备测试视频"""
    video_path = "/Users/yuanliang/Downloads/p.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"❌ 测试视频不存在: {video_path}")
        return None
    
    file_size = os.path.getsize(video_path)
    logger.info(f"✅ 测试视频信息:")
    logger.info(f"   路径: {video_path}")
    logger.info(f"   大小: {file_size / (1024*1024):.2f} MB")
    
    # 创建测试工作目录
    test_dir = project_root / "video_test_workspace"
    test_dir.mkdir(exist_ok=True)
    
    # 复制视频到工作目录
    test_video = test_dir / "test_video.mp4"
    if not test_video.exists():
        logger.info("🔄 复制测试视频到工作目录...")
        shutil.copy2(video_path, test_video)
        logger.info(f"✅ 视频已复制到: {test_video}")
    
    return str(test_video)

def test_step010_demucs(video_path):
    """测试人声分离"""
    logger.info("🎵 [Step 010] 开始人声分离测试...")
    
    try:
        from tools.step010_demucs_vr import separate_all_audio_under_folder
        
        video_dir = Path(video_path).parent
        audio_dir = video_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # 复制视频文件
        import shutil
        video_copy = audio_dir / "video.mp4"
        if not video_copy.exists():
            shutil.copy2(video_path, video_copy)
        
        logger.info("🔄 执行人声分离...")
        start_time = time.time()
        result, vocal_path, instrumental_path = separate_all_audio_under_folder(str(video_dir))
        end_time = time.time()
        
        logger.info(f"✅ 人声分离完成，耗时: {end_time - start_time:.2f}秒")
        logger.info(f"   结果: {result}")
        logger.info(f"   人声轨道: {vocal_path}")
        logger.info(f"   背景音轨道: {instrumental_path}")
        
        # 验证输出文件
        if vocal_path and os.path.exists(vocal_path):
            vocal_size = os.path.getsize(vocal_path)
            logger.info(f"✅ 人声文件大小: {vocal_size / (1024*1024):.2f} MB")
        else:
            logger.error("❌ 人声文件未生成")
            return None, None
            
        if instrumental_path and os.path.exists(instrumental_path):
            inst_size = os.path.getsize(instrumental_path)
            logger.info(f"✅ 背景音文件大小: {inst_size / (1024*1024):.2f} MB")
        else:
            logger.warning("⚠️ 背景音文件未生成")
            
        return vocal_path, instrumental_path
        
    except Exception as e:
        logger.error(f"❌ 人声分离失败: {e}")
        logger.exception("详细错误信息:")
        return None, None

def test_step021_asr(vocal_path):
    """测试语音识别"""
    logger.info("🎤 [Step 021] 开始语音识别测试...")
    
    try:
        from tools.step021_asr_whisperx import whisperx_transcribe_audio
        
        if not vocal_path or not os.path.exists(vocal_path):
            logger.error("❌ 人声轨道文件不存在")
            return None
            
        logger.info("🔄 执行语音识别...")
        start_time = time.time()
        transcript = whisperx_transcribe_audio(
            wav_path=vocal_path,
            model_name='large',
            device='mps',
            diarization=True
        )
        end_time = time.time()
        
        if transcript:
            logger.info(f"✅ 语音识别完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"   识别到 {len(transcript)} 条字幕")
            
            # 显示前几条字幕示例
            logger.info("📝 字幕示例:")
            for i, item in enumerate(transcript[:3]):
                logger.info(f"   [{i+1}] {item.get('start', 0):.2f}-{item.get('end', 0):.2f}s: {item.get('text', '')}")
                if 'speaker' in item:
                    logger.info(f"       说话人: {item['speaker']}")
            
            # 保存字幕文件
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
        logger.error(f"❌ 语音识别失败: {e}")
        logger.exception("详细错误信息:")
        return None

def test_step035_translation(transcript):
    """测试翻译功能"""
    logger.info("🔤 [Step 035] 开始翻译测试...")
    
    try:
        from tools.step035_translation_qwen import get_llm_api_config, llm_response
        
        # 检查配置
        api_key, base_url, model_name = get_llm_api_config()
        logger.info(f"✅ 当前LLM配置:")
        logger.info(f"   模型: {model_name}")
        logger.info(f"   Base URL: {base_url}")
        logger.info(f"   API Key: {api_key[:10]}...")
        
        if not transcript:
            logger.error("❌ 字幕数据为空")
            return None
            
        # 取前5句进行翻译测试
        texts = [item['text'] for item in transcript[:5] if item.get('text', '').strip()]
        if not texts:
            logger.error("❌ 没有可翻译的文本")
            return None
            
        text_batch = "\n".join(texts)
        logger.info(f"🔄 翻译 {len(texts)} 句文本...")
        
        prompt = f"""你是一个专业的视频字幕翻译员。请将以下文本翻译为简体中文，保持语境连贯，不要添加任何额外的解释。

{text_batch}"""
        
        messages = [{"role": "user", "content": prompt}]
        
        start_time = time.time()
        translation = llm_response(messages)
        end_time = time.time()
        
        if translation:
            logger.info(f"✅ 翻译完成，耗时: {end_time - start_time:.2f}秒")
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
        logger.error(f"❌ 翻译失败: {e}")
        logger.exception("详细错误信息:")
        return None

def test_step045_tts(transcript, vocal_path):
    """测试情绪配音"""
    logger.info("🎙️ [Step 045] 开始情绪配音测试...")
    
    try:
        from tools.step045_tts_cinecast import generate_tts_with_emotion_clone
        
        if not transcript or not vocal_path:
            logger.error("❌ 缺少必要数据")
            return False
            
        output_dir = Path(vocal_path).parent.parent / "dubbing_output"
        output_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = min(len(transcript), 3)  # 测试前3句
        
        logger.info(f"🔄 生成 {total_count} 句配音...")
        
        for i, item in enumerate(transcript[:total_count]):
            text = item.get('text', '').strip()
            start_time = item.get('start', 0)
            end_time = item.get('end', start_time + 3)
            
            if not text:
                continue
                
            output_file = output_dir / f"dub_{i:04d}.mp3"
            
            logger.info(f"🔊 生成第{i+1}句: '{text[:30]}...' ({start_time:.2f}-{end_time:.2f}s)")
            
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
                logger.info(f"✅ 第{i+1}句配音完成 ({file_size} 字节)")
                success_count += 1
            else:
                logger.error(f"❌ 第{i+1}句配音失败")
        
        logger.info(f"🏁 情绪配音完成: {success_count}/{total_count} 句成功")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ 情绪配音失败: {e}")
        logger.exception("详细错误信息:")
        return False

def generate_final_report(test_results):
    """生成最终测试报告"""
    logger.info("📋 生成测试报告...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_file": "/Users/yuanliang/Downloads/p.mp4",
        "test_results": test_results,
        "summary": {}
    }
    
    # 计算成功率
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    report["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": f"{success_rate:.1f}%"
    }
    
    # 保存报告
    report_file = project_root / "video_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info("🎉 测试完成!")
    logger.info("📊 测试结果汇总:")
    for step, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {step}: {status}")
    
    logger.info(f"📈 总体成功率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    logger.info(f"📄 详细报告已保存到: {report_file}")
    
    return report

def main():
    """主测试流程"""
    setup_logging()
    
    test_results = {}
    
    # 1. 环境检查
    test_results['environment'] = check_environment()
    if not test_results['environment']:
        logger.error("❌ 环境检查失败，测试终止")
        generate_final_report(test_results)
        return
    
    # 2. 准备测试视频
    video_path = prepare_test_video()
    test_results['video_preparation'] = bool(video_path)
    if not video_path:
        logger.error("❌ 视频准备失败，测试终止")
        generate_final_report(test_results)
        return
    
    # 3. 人声分离
    vocal_path, instrumental_path = test_step010_demucs(video_path)
    test_results['demucs_separation'] = bool(vocal_path)
    
    # 4. 语音识别
    transcript = test_step021_asr(vocal_path)
    test_results['speech_recognition'] = bool(transcript)
    
    # 5. 翻译测试
    translation = test_step035_translation(transcript)
    test_results['translation'] = bool(translation)
    
    # 6. 情绪配音测试
    tts_success = test_step045_tts(transcript, vocal_path)
    test_results['emotion_dubbing'] = tts_success
    
    # 7. 生成最终报告
    generate_final_report(test_results)

if __name__ == "__main__":
    main()
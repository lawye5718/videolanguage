#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoLanguage 快速概念验证测试
跳过耗时的人声分离，直接测试核心AI功能
"""

import os
import sys
import time
import json
from pathlib import Path
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add("quick_demo.log", rotation="100 MB", level="DEBUG")

def test_translation_only():
    """测试翻译功能（无需音频处理）"""
    logger.info("🔤 [测试1] AI翻译功能测试...")
    
    try:
        from tools.step035_translation_qwen import get_llm_api_config, llm_response
        
        # 检查LLM配置
        api_key, base_url, model_name = get_llm_api_config()
        logger.info(f"✅ 当前LLM配置:")
        logger.info(f"   模型: {model_name}")
        logger.info(f"   Base URL: {base_url}")
        
        # 测试文本
        test_text = "Hello, welcome to our amazing video processing system. This is a demonstration of AI-powered translation capabilities."
        
        logger.info(f"🔄 翻译测试文本...")
        logger.info(f"   原文: {test_text}")
        
        prompt = f"""你是一个专业的视频字幕翻译员。请将以下英文文本翻译为简体中文，保持语境连贯：

{test_text}"""
        
        messages = [{"role": "user", "content": prompt}]
        
        start_time = time.time()
        translation = llm_response(messages)
        end_time = time.time()
        
        if translation:
            logger.success(f"✅ 翻译测试成功! 耗时: {end_time - start_time:.2f}秒")
            logger.info(f"   译文: {translation}")
            return True
        else:
            logger.error("❌ 翻译返回空结果")
            return False
            
    except Exception as e:
        logger.error(f"❌ 翻译测试出错: {e}")
        logger.exception("详细错误信息:")
        return False

def test_cinecast_tts_only():
    """测试Cinecast TTS功能"""
    logger.info("🎙️ [测试2] Cinecast TTS功能测试...")
    
    try:
        import requests
        
        # 检查API连接
        health_check = requests.get("http://localhost:8888/health", timeout=5)
        if health_check.status_code != 200:
            logger.error("❌ Cinecast API不可用")
            return False
        
        health_data = health_check.json()
        logger.info(f"✅ Cinecast API状态: {health_data.get('status', 'unknown')}")
        
        # 测试TTS生成
        tts_payload = {
            "input": "这是一个AI配音测试，验证情绪克隆功能是否正常工作。",
            "model": "qwen3-tts",
            "voice": "aiden",
            "response_format": "mp3",
            "speed": 1.0
        }
        
        logger.info("🔄 生成测试音频...")
        start_time = time.time()
        response = requests.post(
            "http://localhost:8888/v1/audio/speech",
            json=tts_payload,
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            audio_size = len(response.content)
            logger.success(f"✅ TTS测试成功! 耗时: {end_time - start_time:.2f}秒")
            logger.info(f"   音频大小: {audio_size} 字节")
            
            # 保存测试音频
            test_output = project_root / "quick_demo_output"
            test_output.mkdir(exist_ok=True)
            audio_file = test_output / "tts_demo.mp3"
            with open(audio_file, 'wb') as f:
                f.write(response.content)
            logger.info(f"   音频已保存到: {audio_file}")
            
            return True
        else:
            logger.error(f"❌ TTS生成失败: {response.status_code}")
            logger.error(f"   错误详情: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ TTS测试出错: {e}")
        logger.exception("详细错误信息:")
        return False

def test_whisperx_import():
    """测试WhisperX导入"""
    logger.info("🎤 [测试3] WhisperX导入测试...")
    
    try:
        import whisperx
        logger.success("✅ WhisperX导入成功")
        logger.info(f"   版本: {getattr(whisperx, '__version__', '未知')}")
        return True
    except Exception as e:
        logger.error(f"❌ WhisperX导入失败: {e}")
        return False

def test_demucs_import():
    """测试Demucs导入"""
    logger.info("🎵 [测试4] Demucs导入测试...")
    
    try:
        import demucs.separate
        logger.success("✅ Demucs导入成功")
        logger.info(f"   版本: {getattr(demucs, '__version__', '未知')}")
        return True
    except Exception as e:
        logger.error(f"❌ Demucs导入失败: {e}")
        return False

def generate_demo_report(results):
    """生成演示报告"""
    logger.info("📋 生成快速演示报告...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "demo_type": "快速概念验证",
        "tests": {
            "translation": results[0],
            "tts": results[1],
            "whisperx_import": results[2],
            "demucs_import": results[3]
        },
        "summary": {}
    }
    
    # 计算成功率
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    report["summary"] = {
        "total_tests": total_tests,
        "successful_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": f"{success_rate:.1f}%"
    }
    
    # 保存报告
    report_file = project_root / "quick_demo_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.success("🎉 快速概念验证完成!")
    logger.info("📊 测试结果汇总:")
    test_names = ["AI翻译", "Cinecast TTS", "WhisperX导入", "Demucs导入"]
    for i, (result, test_name) in enumerate(zip(results, test_names)):
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"📈 总体成功率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    logger.info(f"📄 详细报告已保存到: {report_file}")
    
    if success_rate >= 50:
        logger.success("🎊 核心AI功能验证通过！系统基本可用")
    else:
        logger.warning("⚠️  需要进一步调试核心功能")
    
    return report

def main():
    """主测试流程"""
    setup_logging()
    logger.success("🚀 VideoLanguage 快速概念验证开始!")
    logger.info("目标: 验证核心AI功能，跳过耗时的音频处理")
    
    results = [False, False, False, False]  # [translation, tts, whisperx, demucs]
    
    # 1. 测试翻译功能
    results[0] = test_translation_only()
    
    # 2. 测试TTS功能
    results[1] = test_cinecast_tts_only()
    
    # 3. 测试WhisperX导入
    results[2] = test_whisperx_import()
    
    # 4. 测试Demucs导入
    results[3] = test_demucs_import()
    
    # 生成报告
    generate_demo_report(results)

if __name__ == "__main__":
    main()
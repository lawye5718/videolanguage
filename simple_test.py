#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoLanguage 简化测试脚本 - 重点测试核心功能
"""

import os
import sys
import json
from pathlib import Path
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志配置"""
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add("simple_test_log.log", rotation="100 MB", level="DEBUG")

def test_qwen_translation():
    """测试Qwen翻译功能"""
    logger.info("🔤 测试Qwen翻译功能...")
    
    try:
        from tools.step035_translation_qwen import get_llm_api_config, llm_response
        
        # 测试配置加载
        api_key, base_url, model_name = get_llm_api_config()
        logger.info(f"✅ Qwen配置加载成功")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Base URL: {base_url}")
        
        # 测试简单翻译
        test_text = "Hello, how are you today?"
        messages = [
            {"role": "user", "content": f"请将以下英文翻译为中文：{test_text}"}
        ]
        
        response = llm_response(messages)
        if response:
            logger.info("✅ Qwen翻译测试成功")
            logger.info(f"   原文: {test_text}")
            logger.info(f"   译文: {response}")
            return True
        else:
            logger.error("❌ Qwen翻译返回空结果")
            return False
            
    except Exception as e:
        logger.error(f"❌ Qwen翻译测试失败: {e}")
        return False

def test_cinecast_tts():
    """测试Cinecast TTS功能"""
    logger.info("🎙️ 测试Cinecast TTS功能...")
    
    try:
        import requests
        
        # 测试API连接
        health_url = "http://localhost:8888/health"
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("✅ Cinecast API连接正常")
            else:
                logger.error("❌ Cinecast API健康检查失败")
                return False
        except Exception as e:
            logger.error(f"❌ Cinecast API连接失败: {e}")
            return False
        
        # 测试TTS接口 (OpenAI兼容格式)
        tts_url = "http://localhost:8888/v1/audio/speech"
        test_payload = {
            "model": "qwen3-tts",
            "input": "这是一个测试句子。",
            "voice": "aiden",
            "response_format": "mp3",
            "speed": 1.0
        }
        
        try:
            response = requests.post(tts_url, json=test_payload, timeout=30)
            if response.status_code == 200:
                logger.info("✅ Cinecast TTS接口测试成功")
                logger.info(f"   音频大小: {len(response.content)} 字节")
                return True
            else:
                logger.error(f"❌ Cinecast TTS接口返回错误: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cinecast TTS接口调用失败: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Cinecast TTS测试失败: {e}")
        return False

def test_hf_token():
    """测试HF_TOKEN配置"""
    logger.info("🔑 测试HF_TOKEN配置...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            logger.info("✅ HF_TOKEN配置成功")
            logger.info(f"   Token: {hf_token[:10]}...{hf_token[-4:]}")
            return True
        else:
            logger.warning("⚠️ HF_TOKEN未配置")
            return False
            
    except Exception as e:
        logger.error(f"❌ HF_TOKEN测试失败: {e}")
        return False

def test_file_operations():
    """测试文件操作功能"""
    logger.info("📁 测试文件操作功能...")
    
    try:
        # 测试视频文件访问
        video_path = "/Users/yuanliang/Downloads/p.mp4"
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            logger.info(f"✅ 测试视频文件可访问")
            logger.info(f"   路径: {video_path}")
            logger.info(f"   大小: {file_size / (1024*1024):.2f} MB")
        else:
            logger.error("❌ 测试视频文件不存在")
            return False
        
        # 测试输出目录创建
        output_dir = project_root / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        test_file = output_dir / "test.txt"
        with open(test_file, 'w') as f:
            f.write("测试文件")
        
        if test_file.exists():
            logger.info("✅ 文件读写功能正常")
            test_file.unlink()  # 清理测试文件
            return True
        else:
            logger.error("❌ 文件写入失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 文件操作测试失败: {e}")
        return False

def test_environment():
    """测试环境配置"""
    logger.info("🔧 测试环境配置...")
    
    checks = []
    
    # 检查Python版本
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"✅ Python版本: {python_version}")
    checks.append(True)
    
    # 检查关键依赖
    dependencies = ['requests', 'numpy', 'loguru']
    for dep in dependencies:
        try:
            __import__(dep)
            logger.info(f"✅ {dep} 可用")
            checks.append(True)
        except ImportError:
            logger.warning(f"⚠️ {dep} 未安装")
            checks.append(False)
    
    # 检查FFmpeg
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ FFmpeg 可用")
            checks.append(True)
        else:
            logger.warning("⚠️ FFmpeg 不可用")
            checks.append(False)
    except Exception:
        logger.warning("⚠️ FFmpeg 未安装或不可访问")
        checks.append(False)
    
    return all(checks)

def main():
    """主测试流程"""
    setup_logging()
    logger.info("🚀 开始 VideoLanguage 简化功能测试")
    logger.info(f"测试视频: /Users/yuanliang/Downloads/p.mp4")
    
    # 存储测试结果
    results = {}
    
    # 1. 环境测试
    results['environment'] = test_environment()
    
    # 2. HF_TOKEN测试
    results['hf_token'] = test_hf_token()
    
    # 3. 文件操作测试
    results['file_ops'] = test_file_operations()
    
    # 4. Qwen翻译测试
    results['qwen_translation'] = test_qwen_translation()
    
    # 5. Cinecast TTS测试
    results['cinecast_tts'] = test_cinecast_tts()
    
    # 输出测试总结
    logger.info("🎉 测试完成!")
    logger.info("📊 测试结果汇总:")
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {test_name}: {status}")
    
    # 计算总体成功率
    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = (passed_tests / total_tests) * 100
    
    logger.info(f"📈 总体成功率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    # 生成测试报告
    report = {
        "timestamp": "2026-02-28 04:45:00",
        "video_file": "/Users/yuanliang/Downloads/p.mp4",
        "results": results,
        "success_rate": f"{success_rate:.1f}%"
    }
    
    report_file = project_root / "test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📄 测试报告已保存到: {report_file}")

if __name__ == "__main__":
    main()
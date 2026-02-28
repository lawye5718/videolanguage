# VideoLanguage 项目配置说明

## 🎯 核心配置集成

### 1. Qwen 大模型翻译配置（已集成）

**特点**：自动复用 Cinecast 项目的 Qwen API 配置

**配置来源**：
- 优先读取：`../cinecast/qwen_api_config.json`
- 备用方案：环境变量 `QWEN_API_KEY`

**当前配置状态**：
✅ 成功加载 Cinecast 配置文件
- API Key: sk-44bbcae...422b  
- Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
- Model: qwen3.5-plus

**注意事项**：
⚠️ 当前API配额已用完，需要充值或更换密钥

### 2. HuggingFace Token 配置（说话人分离）

**用途**：启用 WhisperX 的说话人分离功能（Speaker Diarization）

**获取步骤**：
1. 访问 [Hugging Face](https://huggingface.co)
2. 申请访问以下模型的权限：
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. 在 Settings -> Access Tokens 创建新Token
4. 复制以 `hf_` 开头的Token

**配置方法**：
在项目根目录创建 `.env` 文件：
```env
HF_TOKEN=hf_xxxxxxxxx_your_actual_token_xxxxxxxxx
```

### 3. 环境变量配置模板

参考 `.env.example` 文件创建您的配置：

```env
# Qwen API Configuration (自动从cinecast加载，无需手动配置)
# QWEN_API_KEY=your_qwen_api_key_here
# QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# Hugging Face Token for Speaker Diarization
HF_TOKEN=hf_xxxxxxxxx_your_actual_token_xxxxxxxxx

# Model Configuration
QWEN_MODEL_ID=qwen3.5-plus
```

## 🛠️ 功能模块状态

### ✅ 已完成集成
- [x] Qwen翻译模块 - 复用Cinecast配置
- [x] WhisperX语音识别 - MPS优化
- [x] 智能情绪配音 - 与Cinecast TTS集成
- [x] 内存管理优化 - Mac统一内存保护

### 🔄 运行状态
- **翻译功能**：配置正常，等待API配额恢复
- **语音识别**：需要安装whisperx依赖
- **说话人分离**：需要HF_TOKEN配置
- **情绪配音**：功能完整，可正常使用

## 📋 使用建议

### 开发环境准备
```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装音频处理依赖
brew install ffmpeg

# 安装额外工具
pip install python-dotenv openai audiostretchy
```

### 测试流程
1. 验证Qwen配置加载
2. 配置HF_TOKEN启用说话人分离
3. 测试情绪配音功能
4. 运行完整视频处理流水线

## ⚠️ 注意事项

1. **API配额管理**：监控Qwen API使用情况，及时充值
2. **Mac兼容性**：MPS在某些Pyannote算子上可能不稳定，可降级到CPU
3. **内存管理**：长时间批处理建议启用定期内存清理
4. **模型下载**：首次运行WhisperX会自动下载模型，请保持网络畅通

## 🆘 故障排除

### 常见问题
- **ModuleNotFoundError**: 缺少依赖包，使用pip安装
- **API配额不足**: 更换API密钥或充值
- **MPS算子错误**: 将device改为"cpu"
- **说话人分离失败**: 检查HF_TOKEN配置和网络连接
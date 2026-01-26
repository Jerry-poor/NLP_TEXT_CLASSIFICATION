# DDC Text Classification Pipeline

## 目录结构

```
model/API/
├── .env                    # API 密钥配置
├── .env.example            # 配置示例
├── DDClabel_deepseek_hierarchical.csv  # DDC 标签释义表
├── main_evaluation.py      # 批量评估入口脚本
├── README.md
├── data_pipeline/          # 核心流水线模块
│   ├── config.py           # 配置与层级管理
│   ├── data_loader.py      # 数据加载与预处理
│   ├── generator.py        # Prompt 构建
│   ├── inference.py        # API 推理
│   ├── validator.py        # 响应解析
│   ├── metrics.py          # 指标计算
│   ├── orchestrator.py     # 流程编排
│   └── reporting.py        # 报告生成
├── datasets/               # 数据集目录
│   ├── wos_unified.csv
│   ├── ag_news_unified.csv
│   └── lib_unified.csv
├── result/                 # 评估结果（按模型分目录）
└── old/                    # 历史代码归档
```

## 快速开始

### 1. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY_1
```

### 2. 单数据集评估
```bash
python -m data_pipeline.orchestrator --dataset wos_unified.csv --level 2 --sample_size 500 --shot_type few --model deepseek-chat
```

### 3. 批量评估（推荐）
```bash
python main_evaluation.py --model deepseek-chat --sample_size 500
```

自动评估：
- WOS: Level 1, 2
- AG News: Level 1, 2
- Lib: Level 1, 2, 3

结果保存至 `result/<model_name>/`

## 技术规格

| 参数 | 值 |
|------|-----|
| Random Seed | 42 |
| Temperature | 0.0 |
| Max Tokens | 200 |
| Workers | 20 |
| Output Format | JSON |

## 注意事项

1. **WOS 数据集**不支持 Level 3 评估
2. 评估采用 **Blind Evaluation**（Prompt 中不暴露 DDC Code）
3. 无效标签已从释义表中清洗

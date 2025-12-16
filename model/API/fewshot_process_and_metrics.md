## Few-shot 三级分类处理流程与统计说明

### 数据与采样
- Level1/2/3 数据集分别位于 `model/API/dataset/`：
  - Level1: `Lib_Dataset_Level1_30Nov25_final.xlsx`（列：`DDC-L1`, `Title`, `Abstract`）
  - Level2: `Lib_Dataset_Level2_27Nov25_final.xlsx`（列：`DDC-L2`, `Title`, `Abstract`）
  - Level3: `Lib_Dataset_Level3_26Nov25_final.xlsx`（列：`DDC-L3`, `Title`, `Abstract`）
- 每次运行固定 3-shot 支持样本：使用数据集的前 3 行；评估样本从第 4 行开始，按 `--sample_size` 随机抽取（示例运行为 5 条）。
- 标签名称映射来自 `model/API/dewey_decimal_unique.csv`，数字代码仅用于内部 bucket 和解析，模型只看到文本标签。

### 候选构造（最多 10 个）
- Level1：固定 10 个一级代码（000–900），过滤掉映射缺失/Unknown 项。
- Level2：与目标真值同一级桶（如 430 → 400–490，每隔 10 一个，共 10 个），缺失项过滤。
- Level3：与目标真值同十位桶（如 433 → 430–439，共 10 个），缺失项过滤。

### Prompt 结构（`method1/multi_level_few_shot.py`）
- Allowed categories：仅文本标签列表（不含数字）。
- Support examples：3 条示例，展示标题、摘要、真值标签（以 `### 标签 ###` 标记）。
- 任务：从 Allowed 中输出 1–3 个标签名称及置信度，不输出数字码。

### 返回解析与映射
- 模型返回的类别字符串去掉置信度后，与 Allowed 名称做匹配（精确/前缀/包含），解析回代码；若解析失败则记为空。
- 保存字段包含 `predicted_categories_with_codes`（原始返回、置信度、解析出的代码）、`predicted_code`、`predicted_name`。

### 统计指标
- 针对评估样本（不含前 3 条支持）计算 Top-1/3/5：
  - 比较解析得到的代码列表与真实代码（`true_level{1|2|3}_code`）。
  - 统计存入同名 `.md` 摘要。

### 示例运行（各 5 样本）
- Level1：`classification_results/level1_fewshot_sample5.csv` / `.md`
  - Top-1: 40% (2/5)，Top-3/5: 80% (4/5)
- Level2：`classification_results/level2_fewshot_sample5.csv` / `.md`
  - Top-1: 40% (2/5)，Top-3/5: 80% (4/5)
- Level3：`classification_results/level3_fewshot_sample5.csv` / `.md`
  - Top-1: 80% (4/5)，Top-3/5: 100% (5/5)

### 使用命令（示例，深度求索模型）
```bash
cd model/API
python method1/multi_level_few_shot.py --level 1 --input dataset/Lib_Dataset_Level1_30Nov25_final.xlsx --output classification_results/level1_fewshot_sample5.csv --sample_size 5 --provider deepseek-chat --model deepseek-chat
python method1/multi_level_few_shot.py --level 2 --input dataset/Lib_Dataset_Level2_27Nov25_final.xlsx --output classification_results/level2_fewshot_sample5.csv --sample_size 5 --provider deepseek-chat --model deepseek-chat
python method1/multi_level_few_shot.py --level 3 --input dataset/Lib_Dataset_Level3_26Nov25_final.xlsx --output classification_results/level3_fewshot_sample5.csv --sample_size 5 --provider deepseek-chat --model deepseek-chat
```

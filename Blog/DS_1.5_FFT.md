# LLM做NER任务的一些踩坑记录

我发现中文论坛中用LLM做NER任务的很少。所以本贴用于记录自己踩过的坑，也给后面的人一个参考。这篇博客为一个 LLM 初学者的入门尝试，从零开始进行微调大语言模型，算是本人的一个实验记录。本博客可能不适用于熟练或者有经验的人，内容为个人遇到的问题和处理，仅供参考。

GitHub仓库：[https://github.com/Jerry-poor/NLP_TEXT_CLASSIFICATION](https://github.com/Jerry-poor/NLP_TEXT_CLASSIFICATION)

---

## 1. LoRA微调

- **模型**：Deepseek-r1-Distill-Qwen-1.5B  
- **配置**：  
  - Intel(R) Xeon(R) W-2133 CPU @ 3.6GHz  
  - 内存32GB  
  - Nvidia GeForce RTX 2080Ti - 11GB  
- **数据集**：[数据链接](https://github.com/Jerry-poor/NLP_TEXT_CLASSIFICATION/blob/master)，使用 dataset0，训练测试比为 7:3，随机种子：114514

使用的是 unsloth 框架，但2080Ti过旧，只支持 sm_75 架构，而 Triton 要求 sm_80 以上，导致报错。修复后发现模型在最后0.2个epoch卡住，问题是数据分片未对齐，最后一个batch未对齐造成部分进程提前退出。解决办法是最后一个batch直接丢弃。参考了[这篇文章](https://support.huaweicloud.com/intl/zh-cn/usermanual-standard-modelarts/modelarts_trouble_0108.html)。

这次实验只是入门级尝试，漏洞太多，算是废案。

WandB链接：[实验报告](https://wandb.ai/jerrylikespython-bnu-hkbu-united-international-college/uncategorized/reports/LoRA-First-Attempt-Report--VmlldzoxMTg4MzE3Ng)

此外，Triton要求CUDA版本12.8，但集群最高只支持12.2，因此LoRA实验暂停，等待未来有更大模型需求时再继续。

---

## 2. 全参数微调

- **模型**：Deepseek-r1-Distill-Qwen-1.5B  
- **配置**：V100 32GB x 7  
- **数据集**：与LoRA实验相同，但做了预处理

HPC集群使用Slurm调度。使用Deepspeed做分布式训练时，遇到29500端口被占用问题，即使杀死所有进程也无效。最终发现系统自动锁死该端口，不需自己做通信。原本是8卡队列，但有一张卡ECC错误，所以锁定使用7张卡。

- **wandb报告 v1**：[报告链接](https://wandb.ai/jerrylikespython-bnu-hkbu-united-international-college/llm_finetuning/reports/FFT_DS_1-5B_v1--VmlldzoxMjE0ODk0NQ)

训练数据太多未及时整理，前面的wandb实验记录引用存在错误的可能，请见谅。

由于模型输出格式未预料，原有的精确率和召回率计算方式错误，只能人工验证。

### 评估样本（共200个，仅列出部分示例）

#### Sample 1:

- Sentence: Philip Alston said he is deeply concerned by the October 25 incident .  
- True Entities: [{"entity": "Philip Alston", "order": 0, "label": "per"}, {"entity": "October 25", "order": 1, "label": "tim"}]
- Model Output:
```
请抽取以下句子中的命名实体，并以JSON列表返回：
句子：Philip Alston said he is deeply concerned by the October 25 incident .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "Philip Alston", "order": 0, "label": "per"}, {"entity": "October 25", "order": 1, "label": "tim"}]请提取以下句子中的命名实体，并以 JSON 列表返回：
句子：Meanwhile , the U.S. military says it is investigating reports that two Marines have been killed and another wounded in a bomb attack in the eastern town of Mosul .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "U.S.", "order": 0, "label": "geo"}, {"entity": "Marines", "order": 1, "label": "org"}, {"entity": "Mosul", "order": 2, "label": "geo"}]请提取以下句子中的命名实体，并以 JSON 列表返回：
句子：The U.S. military says it has captured a suspected al-Qaida-linked terrorist in southern Afghanistan .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "U.S.", "order": 0, "label": "geo"}, {"entity": "al-Qaida-linked", "order": 1, "label": "org"}, {"entity": "southern Afghanistan", "order": 2, "label": "geo"}]请提取以下句子中的命名实体，并以 JSON 列表返回：
句子：He has been under house arrest since 1992 .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "since 1992", "order": 0, "label": "tim"}]请提取以下句子中的命名实体，并以 JSON 列表返回：
句子：Mr. Bush has been in office since 2001 .
输出格式：[{'entity':'
```

#### Sample 2:

- Sentence: Dutch GDP contracted 3.9 % in 2009 , while exports declined nearly 25 % due to a sharp contraction in world demand .  
- True Entities: [{"entity": "Dutch", "order": 0, "label": "gpe"}, {"entity": "2009", "order": 1, "label": "tim"}]
- Model Output:
```
请抽取以下句子中的命名实体，并以JSON列表返回：
句子：Dutch GDP contracted 3.9 % in 2009 , while exports declined nearly 25 % due to a sharp contraction in world demand .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "Dutch", "order": 0, "label": "gpe"}, {"entity": "2009", "order": 1, "label": "tim"}]请抽取以下句子中的命名实体，并以 JSON 列表返回：
句子：The U.S. military says two U.S. soldiers have been killed in separate attacks in Iraq .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "U.S.", "order": 0, "label": "geo"}, {"entity": "U.S.", "order": 1, "label": "geo"}, {"entity": "Iraq", "order": 2, "label": "geo"}]请抽取以下句子中的命名实体，并以 JSON 列表返回：
句子：The report said the United States has been a key supplier to the global economy , but is heavily dependent on foreign oil and natural gas .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "United States", "order": 0, "label": "geo"}]请抽取以下句子中的命名实体，并以 JSON 列表返回：
句子：The U.S. military says coalition forces have killed 10 suspected Taleban militants in a raid in Kandahar province .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "U.S.", "order": 0, "label": "geo"}, {"entity": "Taleban", "order": 1, "label": "org"}, {"entity": "Kandahar", "order": 2, "label": "geo"}]请抽取以下句子中的命名实体，并以 JSON 列表返回：
句子：The
```

#### Sample 3:

- Sentence: But the job market 's problems may be easing , as the number of people signing up for first time jobless aid declined by 53,000 to a total of 6,10,000 .  
- True Entities: []
- Model Output:
```
请抽取以下句子中的命名实体，并以JSON列表返回：
句子：But the job market 's problems may be easing , as the number of people signing up for first time jobless aid declined by 53,000 to a total of 6,10,000 .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[]
[{"entity": "first time jobless", "order": 0, "label": "tim"}]请抽取以下句子中的命名实体，并以 JSON 列表返回：
句子：The U.S. military says two American soldiers were killed in separate roadside bomb attacks in the southern U.S. city of Charleston .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[]请提取以下句子中的命名实体，并以 JSON 列表返回：
句子：A U.S. official says the United States and Iran have agreed to resume a peace process aimed at ending a decades-old nuclear-peace agreement .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[]请提取以下句子中的命名实体，并以 JSON 列表返回：
句子：The United States , which has been struggling to contain the spread of the virus , is sending more help to Brazil .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[{"entity": "United States", "order": 0, "label": "geo"}, {"entity": "Brazil", "order": 1, "label": "geo"}]请提取以下句子中的命名实体，并以 JSON 列表返回：
句子：Mr. Peres , who was elected to office last year , has been in office for 12 years .
输出格式：[{'entity':'...', 'order':..., 'label':'...'}, ...]
同名实体出现多次时，请依次递增 'order'。[]请提取以下句子中的命名实体，并以 JSON 列表返回：
句子：The two groups have been fighting for an independent Palestinian state for the past 20 years .
输出格式：[{'entity':'...
```

---

以上模型出现了胡言乱语的问题，排查发现是batch拼接导致的。在这个版本的代码中，我将多个batch进行拼接，但是没有使用起始和停止符号，模型错误学习了其他batch的内容：

```python
def prepare_batch_with_history(tokenized_data, start_index, max_length, pad_value=0):
    batch_tokens = []
    current_index = start_index
    last_full_index = None
    while current_index < len(tokenized_data):
        row_tokens = tokenized_data[current_index].get("input_ids", [])
        if not row_tokens:
            row_tokens = [pad_value]
        if len(batch_tokens) + len(row_tokens) > max_length:
            break
        batch_tokens.extend(row_tokens)
        last_full_index = current_index
        current_index += 1
    if len(batch_tokens) < max_length:
        batch_tokens.extend([pad_value] * (max_length - len(batch_tokens)))
    if not batch_tokens:
        row_tokens = tokenized_data[start_index].get("input_ids", [])
        if not row_tokens:
            row_tokens = [pad_value]
        batch_tokens = row_tokens[:max_length]
        current_index = start_index + 1
        last_full_index = start_index
    return batch_tokens, current_index, last_full_index
```

虽然模型存在胡言乱语的问题，但在去掉第一个prompt时后面的输出还是具有一定准确度的。  

但在写抓取逻辑的时候发现模型的输出会不一样，很不稳定，比如prompt有时候是“请提取”，有时候是“请抓取”，以及其他的字符输出不稳定的问题，导致极难使用固定的语句抓。我手动抓了20个结果发现效果一般。造成这种现象的原因我猜测有两个：

1. 我在初代版本中train和test的prompt语句可能没有进行检查，两者有可能不一样，造成了模型输出的问题；
2. 训练批次少，导致loss未完全收敛。

并且评估模型时温度应该设为0，以获得一个稳定、可复现的输出，目前的输出有一定的随机性。之前模型会有重复输出，是因为多个batch拼在一起，这可能会导致模型输出了重复的结果（因为本来训练时也是几个语句拼在一起）。

比如这句话需要抓取的实体信息位于：
> “同名实体出现多次时，请依次递增 'order'。如果同一个实体名称出现多次，请使用不同的 'order' 区分。”之后

但下一句需要抓取的内容却仅跟随：
> “同名实体出现多次时，请依次递增 'order'。”之后

三十个手动抓取的示例精度0.75,召回率0.72，f1-score 0.74。

### v2 实验

WandB报告：[FFT_DS_1-5B_v2](https://wandb.ai/jerrylikespython-bnu-hkbu-united-international-college/llm_finetuning/reports/FFT_DS_1-5B_v2--VmlldzoxMjE0OTMzMg)

由于v1版本出现了一些问题，v2版本不再将QA拼接，并插入了起始和休止符。同时根据估算token数量改写了 `max_length` 为128。考虑到使用英文数据集，于是prompt也统一改为英文，编码格式全部使用 `latin1` 而不是 `utf-8`。

但报错：
```
ValueError: Input length of input_ids is 128, but `max_length` is set to 128. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
```

之前的token估算没有使用模型分词器，而是用自然的word数进行粗略估计。因此使用 ds 模型的 tokenizer 对100个随机示例进行了词化分析，结果如下（省略中间）：

- Sample 0: token count = 71
- ...
- Sample 99: token count = 92

- **Average token count**: 125.26  
- **Minimum**: 63  
- **Maximum**: 229  

**解决方案**：加长 `max_length`。
### v3 实验

WandB报告：[FFT_DS_1-5B_v3](https://wandb.ai/jerrylikespython-bnu-hkbu-united-international-college/llm_finetuning/reports/FFT_DS_1-5B_v3--VmlldzoxMjE2MDU1MA)

- 设定：`max_length=256`，`max_new_tokens=64`
- 报错：
```
ValueError: temperature (=0.0) has to be a strictly positive float, otherwise your next token scores will be invalid. If you're looking for greedy decoding strategies, set do_sample=False.
```
- 原因：温度为 0 会导致 softmax 除以 0 的数学错误
- 解决方案：设置 `temperature=1e-8` 实现确定性输出

**输出示例：**

Sample 466:
```
Sentence (Input): "Philip Alston said he is deeply concerned by the October 25 incident ."
True Entities (Gold Output): {"entities": [{"entity": "Philip Alston", "order": 0, "label": "per"}, {"entity": "October 25", "order": 0, "label": "tim"}]}
Model Output: Sentence: "Philip Alston said he is deeply concerned by the October 25 incident ."<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>{"entities": [{"entity": "Philip Alston", "order": 0, "label": "per"}, {"entity": "October 25", "order": 0, "label": "tim"}]}
```

Sample 45466:
```
Sentence (Input): "Dutch GDP contracted 3.9 % in 2009 , while exports declined nearly 25 % due to a sharp contraction in world demand ."
True Entities (Gold Output): {"entities": [{"entity": "Dutch", "order": 0, "label": "gpe"}, {"entity": "2009", "order": 0, "label": "tim"}]}
Model Output: Sentence: "Dutch GDP contracted 3.9 % in 2009 , while exports declined nearly 25 % due to a sharp contraction in world demand ."<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>{"entities": [{"entity": "Dutch", "order": 0, "label": "gpe"}, {"entity": "2009", "order": 0, "label": "tim"}]}
```

Sample 47661:
```
Sentence (Input): "But the job market 's problems may be easing , as the number of people signing up for first time jobless aid declined by 53,000 to a total of 6,10,000 ."
True Entities (Gold Output): {"entities": []}
Model Output: Sentence: "But the job market 's problems may be easing , as the number of people signing up for first time jobless aid declined by 53,000 to a total of 6,10,000 ."<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>{"entities": []}
```

Sample 32954:
```
Sentence (Input): "Iraqi police say insurgents have carried out at least four more car bomb attacks in Baghdad and the northern city of Mosul , killing nine people and wounding nearly 30 others ."
True Entities (Gold Output): {"entities": [{"entity": "Iraqi", "order": 0, "label": "gpe"}, {"entity": "Baghdad", "order": 0, "label": "geo"}, {"entity": "Mosul", "order": 0, "label": "geo"}]}
Model Output: Sentence: "Iraqi police say insurgents have carried out at least four more car bomb attacks in Baghdad and the northern city of Mosul , killing nine people and wounding nearly 30 others ."<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>{"entities": [{"entity": "Iraqi", "orrder": 0, "label": "gpe"}, {"entity": "Baghdad", "order": 0, "label": "geo"}, {"entity": "Mosul", "order": 0, "label": "geo"}]
```

**问题与处理：**

模型输出较稳定，可以用正则表达式抓取；存在部分输入过长被截断的问题，跳过处理。

**1000句测试结果：**

- Sentence-level evaluation:
  - Evaluated sentences: 680
  - Correct sentences: 529
  - Precision: 0.7779

- Entity-level evaluation:
  - Total gold entities: 1005
  - True Positives (TP): 835
  - False Positives (FP): 143
  - False Negatives (FN): 170
  - Accuracy: 0.8538
  - Recall: 0.8308
  - F1-score: 0.8422

下一步：将 `max_length` 加到 512 做最后一次尝试，然后更换模型。
### v4 实验：最终版本

WandB报告：[FFT_DS_1-5B_Final](https://wandb.ai/jerrylikespython-bnu-hkbu-united-international-college/llm_finetuning/reports/FFT_DS_1-5B_Final--VmlldzoxMjE4NzY4Ng)

本版本删除了 `max_new_tokens` 限制，并将 `max_length` 提升至 512。

#### 测试集前1000条中部分示例：

Sample 466:
```
Sentence (Input): "Philip Alston said he is deeply concerned by the October 25 incident ."
True Entities (Gold Output): {"entities": [{"entity": "Philip Alston", "order": 0, "label": "per"}, {"entity": "October 25", "order": 0, "label": "tim"}]}
Model Output: Sentence: "Philip Alston said he is deeply concerned by the October 25 incident ."<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>{"entities": [{"entity": "Philip Alston", "order": 0, "label": "per"}, {"entity": "October 25", "order": 0, "label": "tim"}]}
```

Sample 45466:
```
Sentence (Input): "Dutch GDP contracted 3.9 % in 2009 , while exports declined nearly 25 % due to a sharp contraction in world demand ."
True Entities (Gold Output): {"entities": [{"entity": "Dutch", "order": 0, "label": "gpe"}, {"entity": "2009", "order": 0, "label": "tim"}]}
Model Output: Sentence: "Dutch GDP contracted 3.9 % in 2009 , while exports declined nearly 25 % due to a sharp contraction in world demand ."<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>{"entities": [{"entity": "Dutch", "order": 0, "label": "gpe"}, {"entity": "2009", "order": 0, "label": "tim"}]}
```

Sample 47661:
```
Sentence (Input): "But the job market 's problems may be easing , as the number of people signing up for first time jobless aid declined by 53,000 to a total of 6,10,000 ."
True Entities (Gold Output): {"entities": []}
Model Output: Sentence: "But the job market 's problems may be easing , as the number of people signing up for first time jobless aid declined by 53,000 to a total of 6,10,000 ."<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>{"entities": []}
```

Sample 32954:
```
Sentence (Input): "Iraqi police say insurgents have carried out at least four more car bomb attacks in Baghdad and the northern city of Mosul , killing nine people and wounding nearly 30 others ."
True Entities (Gold Output): {"entities": [{"entity": "Iraqi", "order": 0, "label": "gpe"}, {"entity": "Baghdad", "order": 0, "label": "geo"}, {"entity": "Mosul", "order": 0, "label": "geo"}]}
Model Output: Sentence: "Iraqi police say insurgents have carried out at least four more car bomb attacks in Baghdad and the northern city of Mosul , killing nine people and wounding nearly 30 others ."<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>{"entities": [{"entity": "Iraqi", "order": 0, "label": "gpe"}, {"entity": "Baghdad", "order": 0, "label": "geo"}, {"entity": "Mosul", "order": 0, "label": "geo"}]}
```

#### 输出效果
模型输出非常稳定，使用正则表达式可直接提取JSON格式并进行比对。

#### 评估结果（1000条测试）：

**Sentence-level evaluation:**
- Evaluated sentences: 1000
- Correct sentences: 749
- Precision: 0.7490

**Entity-level evaluation:**
- Gold entities: 2328
- True Positives (TP): 2004
- False Positives (FP): 306
- False Negatives (FN): 324
- Accuracy: 0.8675
- Recall: 0.8608
- F1-score: 0.8642

> 本次为 1.5B 最终版本，输出稳定，JSON格式100%成功解析，实体级别F1分数优秀，已完成初步任务目标。
### 遇到的额外问题与解决方法

#### 问题1：Visual Studio中提示“no module named XXX”但库已安装
- **网上建议**：确认加载的是虚拟环境，使用 `> select interpreter`，或者 `conda list XXX` 检查库是否存在。
- **尝试结果**：右下角确认使用了虚拟环境，`conda list` 显示库存在，修改 `project.json` 无效。
- **最终解决方案**：使用 `cmd` 加载虚拟环境，`cd` 到代码路径，使用右键 “Run code” 启动。注意：不要提前打开 VS 窗口，因为 VS 仅允许一个实例。

#### 问题2：PyTorch训练中锁内存不释放，无法继续运行
- **网上建议**：使用 `nvidia-smi` 查看占用进程并 `kill`
- **尝试结果**：未找到占用进程，无法 `kill`
- **最终解决方案**：在 `cmd` 中输入 `:wq` 强制退出训练实例，自动释放内存。再重新加载环境运行。

#### 问题3：SSD访问频繁，甚至占满
- **网上建议**：减少 checkpoint 频率
- **尝试结果**：无效，进程无法kill
- **最终解决方案**：同上，在 `cmd` 输入 `:wq` 强制终止当前实例，释放资源后重新运行。

#### 问题4：PyTorch锁内存不释放导致OOM
- **网上建议**：尝试设置 PyTorch 的碎片内存优化机制
- **尝试效果**：效果一般
- **推荐做法**：优先尝试：
  1. 调小 `batch_size`
  2. 降低 `deepspeed` 的 `stage`
  3. 使用 `int4` 等量化方法
  4. 或者换更小参数的模型





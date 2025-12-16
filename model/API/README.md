# API Methods Documentation

## Method 1: Language Generation Confidence (Used)
**描述**: 使用模型生成的文本输出来获取置信度。
**原理**: 要求模型在输出分类类别的同时，以文本形式直接输出其对该预测的置信度百分比（非模型真实的 logits/logprobs）。
**现状**: **当前所有的文本分类结果均基于 Method 1 生成。**

## Method 2: External Control & Logprobs (Failed)
**描述**: 尝试通过 API 调用获取内置置信度（Logprobs），并通过外部控制输出类别来达到外置控制采样。
**原理**: 试图限制模型输出特定的类别 Token，获取这些 Token 的 Logprobs，然后通过 Softmax 归一化形成采样后的置信度。
**现状**: **失败**。由于 API 的限制（如无法精确控制所有 Token 的 Logprobs 返回或受限的 Logit Bias 支持），该方法未能成功实施。

## 结论
请注意，目录中的结果文件均来自 **Method 1**。

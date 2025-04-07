# NLP_TEXT_CLASSIFICATION
📌 Project: Text Classification Algorithm This project aims to develop and optimize algorithms for text classification.

选择语言 / Choose Language:
- [中文 README](./Readme/README_zh.md)
- [English README](./Readme/README_en.md)
⭐全量微调实验已有初步结果:
  Entity-Level Accuracy : 0.8538
  Recall: 0.8308
  F1-score: 0.8422
❀这个结果在目前使用的数据集上超越了NuNER,Spacy的三种预训练管道的微调结果
实验尝试在个人wandb账号有可视化的overview
由于设备和环境分离，git仓库推送会远慢于实际进度
github上的日志会比实际进度慢，实验具体数据比较杂乱，未整理后暂时不上传  
由于时间紧张Readme的补档暂时停止
需要注意的改动
>精确度测算代码已经进行重写，需要对位置进行测算，原测算代码更改为eval函数，并进行了封装。作用可以打开.py文件直接查看。
>具体改动后的调用请参考sm0.py，这个是spacy函数的sm预训练管道在dataset0上的调用函数。可以直接复制粘贴到你们的代码再根据模型不同进行修改。

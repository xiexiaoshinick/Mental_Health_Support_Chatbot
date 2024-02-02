<p align="center">
 <img width="200px" src="./assets//logo.png" align="center" alt="GitHub Readme Stats" />
 <h2 align="center">power by InternLM2-Chat</h2>
 <p align="center">它可以提供关于自我成长、情感、爱情、人际关系等方面的心理健康对话，为用户提供非专业的支持和指导。</p>
 <p align="center">
	<img  src="https://img.shields.io/badge/gitHub-%E4%B8%AD%E6%96%87%E5%BF%83%E7%90%86%E5%81%A5%E5%BA%B7%E5%A4%A7%E6%A8%A1%E5%9E%8B-brightgreen" />
 </p>
</p>

---

# Mentalhealth-心理健康大模型

中文心理健康对话大模型是基于InternLM2-7B-Chat微调后中文心理健康对话模型，可针对例如自我成长、情感、爱情问题、人际关系、行为、家庭、治疗、婚姻和职业等方面提供心理健康对话。
模型可以提供共情和非判断性的回应，为用户提供情感支持。
# 模型和线上演示demo
- [模型](https://modelscope.cn/models/xiexiaoshi/Mental_Health_Support_Chatbot/summary)
- [演示demo](https://openxlab.org.cn/apps/detail/xiexiaoshi/Mental_Health_Support_Chatbot)

# 数据源

模型的数据来源主要有2个,共167229轮对话

|数据来源|描述|用途|
|:----|:----|:----|
|[SMILE](https://github.com/qiuhuachuan/smile)|数据集通过 ChatGPT 改写真实的心理互助 QA 为多轮的心理健康支持多轮对话，该数据集SmileChat含有 55,165k 个多轮对话，其对话主题、词汇和篇章语义更加丰富多样，更加符合在长程多轮对话的应用场景。|提供多轮对话的训练素材|
|[PsyQA](https://github.com/thu-coai/PsyQA)|壹心理论坛数据构建了一份高质量的中文心理咨询数据集，提供了丰富的援助策略标注。可用于生成富有援助策略的长咨询文本。|提供长对话的训练素材|

# 对话样例

|形式|输入|输出|
|:----|:----|:----|
|多轮对话|<ul><li>男生告诉女生，他很差，什么是真正的喜欢？喜欢上一个男生，男生对她很好，有一天两人互相说出了心声，但男生告诉女生，他很差，让女生不要喜欢他，女生该怎么办？</li></ul><ul><li>我了解了，我会多关心他并帮助他度过难关的。</li></ul><ul><li>我以前的确没有太了解他的过去，我会尝试和他更多的交流，了解他的故事。</li></ul><ul><li>谢谢你的支持，我真的觉得有人倾听和关心很重要。我会尽量调整自己的心态，努力度过这个难关。</li></ul>|<ul><li>求助者，我能理解你现在的困惑和矛盾感。首先，重要的不是男生说了什么，而是他的行动是否与他的言辞一致。你可以观察他最近是否遇到了什么困难，尝试与他沟通并给予安慰</li></ul><ul><li>另外，你可以了解一下他的过去经历，特别是他的交友圈和朋友是什么样的人。这些信息可以增加你对他的了解，帮助你做出正确的判断。</li><li>还有一种可能是，他可能有自卑感，才会说自己很差。那你对他是怎么看的呢？你的朋友对他有什么评价？他们的看法可以作为你的参考。</li><li>你很勇敢，相信自己，一定会找到适合自己的答案的。加油！</li>|
|长文本|<ul><li>最近不知道为什么就突然间就觉得自己压力非常大。每天的时间也很紧张，也不知道该怎么样，在紧张的生活里抽出来一点时间去放松。</li>|<ul><li>嗯，可以具体说说你的情况吗？是在工作，还是在学习。如果有机会可以去大自然里散步。这样人心情就会变好。如果没有时间，可以晒晒太阳，抽时间睡个好觉，和家里人好好相处。都能让自己的心情缓和。之前看过一个电视剧，女主人公当时心情非常不好，她爸爸安慰了他，她突然有了信心去面对人生了。家人的力量是伟大的。如果我们能花多一点时间去爱家人，爱自己，爱别人身上。我们的心绪就回慢慢平静下来。感恩。佛祖保佑你。我佛慈悲。祝你平安喜乐</li>|

# 训练

模型采用4张80G显存的A100进行LoRA训练，总计优化了5个epoch。使用OpenXlab的开源训练工具Xtuner

```shell
NPROC_PER_NODE=4 xtuner train internlm2_chat_7b_qlora_mentalhealth_e3.py
```

# 效果展示

![demo](https://s3.bmp.ovh/imgs/2024/01/29/dc77b5d92a29aad0.png)

# 相关论文以及引用信息

```shell
@inproceedings{sun-etal-2021-psyqa,
    title = "PsyQA: A Chinese Dataset for Generating Long Counseling Text for Mental Health Support",
    author = "Sun, Hao  and
      Lin, Zhenru  and
      Zheng, Chujie  and
      Liu, Siyang  and
      Huang, Minlie",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2021",
    year = "2021",
}
```

```shell
@misc{qiu2023smile,
      title={SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support},
      author={Huachuan Qiu and Hongliang He and Shuai Zhang and Anqi Li and Zhenzhong Lan},
      year={2023},
      eprint={2305.00450},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

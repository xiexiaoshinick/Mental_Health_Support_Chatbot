import os
import copy
import random
import shutil
import warnings
import traceback
import torch
from torch import nn
import gradio as gr
from openxlab.model import download
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
from utils.gradio_utils import format_cover_html
from transformers.utils import logging
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM
logger = logging.get_logger(__name__)

model_chinese_name = "中文心理健康大模型"
model_avatar="./assets/image.png"
description ="这是一个基于 InternLM2-7B-Chat 微调的中文心理健康对话模型，它可以提供关于自我成长、情感、爱情、人际关系等方面的心理健康对话，为用户提供非专业的支持和指导。" 

suggests = [
    "我今年13岁，我怀疑自己有点心理问题。",
    "我是20年应届毕业生，找不到工作，压力很大，想放弃。",
    "隐瞒自己实际年龄和男友相处了一年，觉得自己好丢脸？实际比男朋友大了七岁，但是骗他只差一岁，结果他看到了我的护照，感觉特别丢脸，不知道该怎么办，想去死来逃避解脱，到底该怎么办？觉得自己隐瞒年纪好恶心，好丢脸！",
    "为什么爱人洗澡，上厕所，吃饭，都要带着手机看视频？"
  ]


customTheme = gr.themes.Default(
    primary_hue=gr.themes.utils.colors.blue,
    radius_size=gr.themes.utils.sizes.radius_none,
)
# async def model_download(model_repo, output):
#     if not os.path.exists(output):
#         await download(model_repo=model_repo, output=output)
#     return output

# model_id = 'xiexiaoshi/Mental_Health_Support_Chatbot'
# model_name_or_path = snapshot_download(model_id, revision='master')

# OpenXLab
model_name_or_path = './Mental_Health_Support_Chatbot'
model_repo = "xiexiaoshi/Mental_Health_Support_Chatbot"
download(model_repo=model_repo,output=model_name_or_path)
# model_name_or_path = '/nfs/volume-379-6/xiewenzhen/xtuner/datas/Tasks/merged'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs["input_ids"]
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]  # noqa: F841  # pylint: disable=W0612
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul((min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break

def clear_history():
    return '', None

user_prompt = "<|im_start|>user\n{user}<|im_end|>\n"
robot_prompt = "<|im_start|>assistant\n{robot}<|im_end|>\n"
cur_query_prompt = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def combine_history(prompt,history):
    meta_instruction = (
        "现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。"
    )
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in history:
        user_message, bot_message = message
        cur_prompt = user_prompt.format(user=user_message)
        total_prompt += cur_prompt
        cur_prompt = robot_prompt.format(robot=bot_message)
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt

def generate_chat(input: str, history = None):
    generation_config = GenerationConfig(max_length=32768, top_p=0.8, temperature=0.7)
    real_prompt = combine_history(input,history)
    output = generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                additional_eos_token_id=92542,
                **asdict(generation_config),
            )
    for x in output:
        history.append((input, x))
        yield None, history
        history.pop()
    history.append((input, x))
    return None, history
# 创建 Gradio 界面
demo = gr.Blocks(css='assets/appBot.css', theme=customTheme)
with demo:
    gr.Markdown(
        '# <center> \N{fire}中文心理健康对话大模型-训练营参赛作品 ([github](https://github.com/xiexiaoshinick/Mental_Health_Support_Chatbot))</center>'  # noqa E501
    )
    draw_seed = random.randint(0, 1000000000)
    state = gr.State({'session_seed': draw_seed})
    with gr.Row(elem_classes='container'):
        with gr.Column(scale=4):
            with gr.Column():
                user_chatbot = gr.Chatbot(
                    elem_id="chatbot", label="Chatbot", height=600)
            with gr.Row():
                with gr.Column(scale=12):
                    preview_chat_input = gr.Textbox(
                        show_label=False,
                        placeholder='和我聊聊天吧！跟我讲讲你的近况和心情，我很愿意倾听～')
                with gr.Column(min_width=70, scale=1):
                    clear_btn = gr.Button(value="🗑️  清除", interactive=True)
                with gr.Column(min_width=70, scale=1):
                    preview_send_button = gr.Button("🚀 发送", variant='primary')

        with gr.Column(scale=1):
            user_chat_bot_cover = gr.HTML(
                format_cover_html(model_chinese_name,description, model_avatar))
            user_chat_bot_suggest = gr.Examples(
                label='Prompt Suggestions',
                examples=suggests,
                inputs=[preview_chat_input])
            
    preview_send_button.click(generate_chat,
               inputs=[preview_chat_input, user_chatbot],
               outputs=[preview_chat_input, user_chatbot])
    clear_btn.click(
        clear_history,
        inputs=[],
        outputs=[user_chatbot, preview_chat_input],
        queue=False)

demo.queue(concurrency_count=10)
demo.launch(show_error=True)

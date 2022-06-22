import os
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_path = "./models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

context = "有四种常见的机器阅读理解任务，分别是完形填空，多项选择，片段抽取，自由问答。"
question = "有哪些机器阅读理解任务？"

inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
outputs = model(**inputs)

answer_start_scores = outputs[0]
answer_end_scores = outputs[1]

answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores)+1

result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])).replace(" ","")

if result in {"[CLS]","[SEP]"}:
    answer = "no answer"
else:
    answer = result

print(answer)




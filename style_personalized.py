from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import torch
from tqdm import tqdm
import numpy as np

# 下载 NLTK 数据（确保运行时已下载）
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

# Dataset
dataset = load_dataset("parquet", data_files={"test": "../data/test-00000-of-00001.parquet"})
test_data_n = dataset['test']  
test_data = test_data_n.select(range(10))  # 选择前 10 条

# Model
device = torch.device("cuda:7")
model_address = "/data/Datasets/Llama-2-7b-chat-hf"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_address)
model = AutoModelForCausalLM.from_pretrained(
    model_address,
    quantization_config=quantization_config,
    device_map={"": 7}
)

# 第一步：使用 LLaMA-2-7B-chat 生成风格无关响应
def generate_style_agnostic_response(inputs, max_new_tokens=350):
    responses = []
    for input_text in inputs:    
        inputs_encoded = tokenizer(input_text, return_tensors="pt").to(device)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs_encoded,
                max_new_tokens=max_new_tokens,  # 与 LongLaMP Review Generation 的平均输出长度接近
                do_sample=False,  # 贪婪解码，与论文一致
                return_dict_in_generate=True
            )
        # response ->  [batch_size, sequence_length]
        # input_text为一个样本，故这里用0。sequences包括输入token+输出token，因此取输入token后面的token
        response = tokenizer.decode(outputs.sequences[0, inputs_encoded['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        responses.append(response)
    return responses

# 第二步：计算 StyleVector
def compute_style_vector(user_history, layer_idx=20):
    positive_activations = []
    negative_activations = []

    # y_positive_activation=[]
    # y_negative_activation=[]

    # all_positive_activation=[]
    # all_negative_activation=[]
    

    for item in user_history:
        historical_input = (
            f"Generate the review text written by a reviewer who has given an overall rating of \"{item['overall']}\" "
            f"for a product with description \"{item['description']}\". The summary of the review text is \"{item['summary']}\""
        )
        authentic_text = item['reviewText']
        style_agnostic_text = generate_style_agnostic_response([historical_input])[0]

        positive_input = historical_input + authentic_text
        negative_input = historical_input + style_agnostic_text

        len_x = len(tokenizer(historical_input, return_tensors="pt")['input_ids'][0])

        model.eval()
        with torch.no_grad():
            inputs = tokenizer(positive_input, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            positive_hidden = outputs.hidden_states[layer_idx][0, -1, :].cpu().numpy()
            positive_activations.append(positive_hidden)

            # positive_all_hidden = outputs.hidden_states[layer_idx][0, :, :].mean(dim=0).cpu().numpy()
            # positive_y_hidden = outputs.hidden_states[layer_idx][0, len_x:, :].mean(dim=0).cpu().numpy()
            # y_positive_activation.append(positive_y_hidden)
            # all_positive_activation.append(positive_all_hidden)


            inputs = tokenizer(negative_input, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            negative_hidden = outputs.hidden_states[layer_idx][0, -1, :].cpu().numpy()
            negative_activations.append(negative_hidden)

            # negative_all_hidden = outputs.hidden_states[layer_idx][0, :, :].mean(dim=0).cpu().numpy()
            # negative_y_hidden = outputs.hidden_states[layer_idx][0, len_x:, :].mean(dim=0).cpu().numpy()
            # y_negative_activation.append(negative_y_hidden)
            # all_negative_activation.append(negative_all_hidden)

    positive_mean = np.mean(positive_activations, axis=0)
    negative_mean = np.mean(negative_activations, axis=0)

    # positive_all_mean = np.mean(all_positive_activation, axis=0)
    # negative_all_mean = np.mean(all_negative_activation, axis=0)
    # positive_y_mean = np.mean(y_positive_activation, axis=0)
    # negative_y_mean = np.mean(y_negative_activation, axis=0)

    style_vector_tensor = torch.tensor(positive_mean - negative_mean, dtype=model.dtype).to(device)

    # style_vector_all_tensor = torch.tensor(positive_all_mean - negative_all_mean, dtype=model.dtype).to(device)
    # style_vector_y_tensor = torch.tensor(positive_y_mean- negative_y_mean, dtype=model.dtype).to(device)

    return style_vector_tensor

# 第三步：使用 StyleVector 生成个性化响应
def generate_with_style_vector(input_text, style_vector, layer_idx=20, alpha=1.0, max_new_tokens=350):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs['input_ids'].shape[1]

    style_vector_tensor = torch.tensor(alpha * style_vector, dtype=model.dtype).to(device)

    def hook(module, input, output):
        # print(5,output[0].size())
        output[0][:, -1, :] += style_vector_tensor.to(output[0].device)
        return output

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True
        )
    handle.remove()
    response = tokenizer.decode(outputs.sequences[0, input_len:], skip_special_tokens=True).strip() 
    return response

# 生成个性化评论
generated_reviews = []
for i, (input_text, ref_output, profile) in enumerate(tqdm(
    zip(test_data['input'], test_data['output'], test_data['profile']),
    total=len(test_data),
    desc="Generating personalized review"
)):
    style_vector = compute_style_vector(profile)
    generated_text = generate_with_style_vector(input_text, style_vector)
    generated_reviews.append(generated_text)

# ROUGE-L
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_l_scores = [scorer.score(ref, gen)['rougeL'].fmeasure for gen, ref in zip(generated_reviews, test_data['output'])]
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

#  METEOR
meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(gen)) for gen, ref in zip(generated_reviews, test_data['output'])]
avg_meteor = sum(meteor_scores) / len(meteor_scores)

print(f"Average ROUGE-L: {avg_rouge_l:.4f}")
print(f"Average METEOR: {avg_meteor:.4f}")

from datasets import Dataset
from tqdm import tqdm
import boto3
import json
from transformers import AutoTokenizer

data_path = "data/summaries_fine_tuning/test.json"
component_to_invoke = "base-Llama-31-8B-Instruct-2025-04-02-14-26-42-899"
endpoint_name = "Llama-31-8B-Instruct-2025-04-02-14-26-42-899-endpoint"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "left"


df = Dataset.from_json(data_path).to_pandas()
sm_rt_client = boto3.client(service_name='sagemaker-runtime')

for idx in tqdm(range(len(df))):
    description = df.iloc[idx]["company_descriptions"]
    text = (f"Extract and summarize key information about the company MAISA. Use only text provided in <<<>>>: <<<{description}>>>."
            f"Do not include any system context, such as 'Based on the provided text...', or any system messages in the response."
            f"Description should be short and informative, up to 5 sentences without markup. "
            f"If it is impossible to extract relevant information return None. Remove context from the response")
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response_model = sm_rt_client.invoke_endpoint(
        EndpointName=endpoint_name,
        InferenceComponentName=component_to_invoke,
        Body=json.dumps(
            {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 150, "temperature": 0.9}
            }
        ),
        ContentType="application/json",
    )

    base_model_reponse = json.loads(response_model["Body"].read().decode("utf8"))[0]["generated_text"]
    print(base_model_reponse.split("assistant")[-1].strip())

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
llm = HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen2.5-1.5B-Instruct',
    task = 'text-generation',
   pipeline_kwargs={
    "temperature": 0.5,
    "max_new_tokens": 256
}
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print (result.content)

# import os
# from dotenv import load_dotenv
# from huggingface_hub import HfApi

# load_dotenv()

# token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# print("Token loaded:", token is not None)
# print("Token starts with:", token[:7] if token else None)

# api = HfApi(token=token)

# print("Logged in as:", api.whoami()["name"])

# print("Testing model access...")

# api.model_info("meta-llama/Meta-Llama-3-8B-Instruct")

# print("✅ Model access confirmed!")

# import os
# from dotenv import load_dotenv
# from transformers import AutoTokenizer

# load_dotenv()

# token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# print("Token loaded:", token is not None)

# tokenizer = AutoTokenizer.from_pretrained(
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     token=token
# )

# print("✅ Tokenizer downloaded successfully!")

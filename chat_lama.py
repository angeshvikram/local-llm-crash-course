from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("PrunaAI/Prajwal3009-unisys_lama2-bnb-4bit-smashed",
                                             trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("Prajwal3009/unisys_lama2")

input_ids = tokenizer("What is the color of prunes?,", return_tensors='pt').to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=216)
tokenizer.decode(outputs[0])
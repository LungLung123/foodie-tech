import gpt_2_simple as gpt2
import os
import requests

model_name = "124M"

if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_name=model_name)

print("Come try our coffee for")

prompt = "Hello World!"

print("Generating...")

gpt2_text = gpt2.generate(sess, model_name=model_name, temperature=1.0, length=50, prefix=prompt, batch_size=20, nsamples=20, include_prefix=True, return_as_list=True)[0]

print(gpt2_text)

import gpt_2_simple as gpt2
import os
import requests

model_name = "124M"

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_name=model_name)

prompt = "Hello World!"

gpt2_text = gpt2.generate(sess, model_name=model_name, length=512, prefix=prompt, batch_size=3, nsamples=6, include_prefix=True, return_as_list=True)[0]

print(gpt2_text)

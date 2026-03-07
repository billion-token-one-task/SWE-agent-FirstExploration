import os
import litellm
base=os.getenv('OPENAI_API_BASE_URL') or os.getenv('OPENAI_API_BASE')
key=os.getenv('OPENAI_API_KEY')
for top_p in [None,1.0]:
    try:
        s=litellm.completion(model='gpt-5.3-codex',messages=[{'role':'user','content':'ok'}],api_base=base,api_key=key,stream=True,top_p=top_p,max_tokens=8)
        chunks=list(s)
        print('top_p',top_p,'chunks',len(chunks),'ok')
    except Exception as e:
        print('top_p',top_p,'ERR',type(e).__name__,str(e)[:160])

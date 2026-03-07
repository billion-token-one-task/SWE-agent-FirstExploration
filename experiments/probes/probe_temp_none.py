import os, litellm
base=os.getenv('OPENAI_API_BASE_URL') or os.getenv('OPENAI_API_BASE')
key=os.getenv('OPENAI_API_KEY')
for temp in [None,1.0]:
    try:
        s=litellm.completion(model='gpt-5.3-codex',messages=[{'role':'user','content':'ok'}],api_base=base,api_key=key,stream=True,temperature=temp,max_tokens=8)
        chunks=list(s)
        print('temp',temp,'chunks',len(chunks),'ok')
    except Exception as e:
        print('temp',temp,'ERR',type(e).__name__,str(e)[:180])

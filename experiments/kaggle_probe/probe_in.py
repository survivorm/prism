import os,json
for root,dirs,files in os.walk("/kaggle/input"):
    print(root,"| D:",dirs[:5],"| F:",files[:8],flush=True)
print("PROBE DONE",flush=True)


#%%
from typing import Any, Dict, List

from pathlib import Path
import json
import inspect
import os
os.environ["OLLAMA_API_KEY"] = "97858a1a8e5a45b68348f9da84ee821d.wqxS_Xk5KAaOisP-I_XfEO9q"  
# We need to set the ollama api key before we import ollama. The API key is only needed 
# when using ollama tools such as web_search and web_fetch. You need your own API key,
# which can be generated for free once you have a user profile at ollama.com
from ollama import Client, web_search, web_fetch
import tiktoken

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

#%% 
################################## Define client #########################################

# First we define the client - namely the server we want to talk to
# local client: client = Client(host = "http://localhost:11434")
client = Client(host = "http://10.160.27.152:11434")

client = Client(host = "http://localhost:11434")

#%%
################################## Test 1 - Interpret data #########################################
# Load .csv files and format to pandas.DataFrame with timestamp index column
folder0285 = Path(r"L:\PG-Nanoteknologi\PROJEKTER\Sensorbaserede arbejdspladsmålinger for partikler\Field studies\Isover_XX\Baseline\Isover 2nd round_week 43_2023\Isover_MA200-0285")
files0285 = sorted(p.resolve() for p in folder0285.glob("*.csv") if p.is_file())
dfs0285 = [pd.read_csv(f,header=0,parse_dates=["Date / time local"]) for f in files0285]
df0285 = pd.concat(dfs0285, ignore_index=True)
df0285 = df0285.set_index("Date / time local").sort_index()

folder0353 = Path(r"L:\PG-Nanoteknologi\PROJEKTER\Sensorbaserede arbejdspladsmålinger for partikler\Field studies\Isover_XX\Baseline\Isover 2nd round_week 43_2023\Isover_MA200-0353")
files0353 = sorted(p.resolve() for p in folder0353.glob("*.csv") if p.is_file())
dfs0353 = [pd.read_csv(f,header=0,parse_dates=["Date / time local"]) for f in files0353]
df0353 = pd.concat(dfs0353, ignore_index=True)
df0353 = df0353.set_index("Date / time local").sort_index()

folder0369 = Path(r"L:\PG-Nanoteknologi\PROJEKTER\Sensorbaserede arbejdspladsmålinger for partikler\Field studies\Isover_XX\Baseline\Isover 2nd round_week 43_2023\Isover_MA200-0369")
files0369 = sorted(p.resolve() for p in folder0369.glob("*.csv") if p.is_file())
dfs0369 = [pd.read_csv(f,header=0,parse_dates=["Date / time local"]) for f in files0369]
df0369 = pd.concat(dfs0369, ignore_index=True)
df0369 = df0369.set_index("Date / time local").sort_index()

def AAE_calc(df):
    """
    Function to calculate the Ångstrøm exponent from aethalometer datasets based on the
    IR vs UV channels.
    """    
    # Wavelengths
    wvl = [375., 470., 528., 625., 880.] # 

    # Specific attenuation cross-section
    sigma = np.array([24.069, 19.070, 17.028, 14.091, 10.120]) # m**2/g
    Cref = 1.3 # Multiple scattering coefficient

    # Mass absorbtion cross section (MAC)
    MAC = sigma/Cref # m**2/g
    
    # Absorption coefficients for UV and IR
    conc_keys = ['UV BCc', 'Blue BCc', 'Green BCc', 'Red BCc']
    abs_880 = df['IR BCc']*10**(-6)*MAC[-1] # m**-1, IR

    for i, key in enumerate(conc_keys):
        b_abs = np.array(df[key])*10**(-6)*MAC[i]

        # Absorption Ångstrøm exponent (AAE)
        AAE = -(np.log(b_abs/abs_880)/np.log(wvl[i]/880.))

        conc = key.split(' ')[0]
        df_key = f'{conc} and IR'
        df[df_key] = AAE

    return df

# Calculate the Ångstrøm exponents and select the UV/IR version
AEE = AAE_calc(df0369)["UV and IR"].resample("5min").median()

csv_text = AEE.to_csv()

# Generate the full input for the model
full_input = [
    {
        "role": "system",
        "content": (
            "You are an expert in aerosol science and atmospheric optics. "
            "You specialize in interpreting Ångström exponents (AAE), aerosol "
            "absorption properties, and aethalometer datasets. "
            "Always base your answer directly on the provided data and context, "
            "and explicitly mention the Ångström exponent in your answer."
        ),
    },
    {
        "role": "user",
        "content": (
            
            "CONTEXT: The data below is from a measurement campaign in a glass wool "
            "production facility. Measurements were taken during cutting of "
            "glass wool mats using a wedge cutter with a MA200 Aethalometer."
            "The index is timestamps; the column is the UV vs IR Ångstrøm exponent.\n\n"
            
            "TASK: Based **only** on this Ångstrøm exponent time series:\n"
            "1. What does the exponent indicate about aerosol composition and likely sources"
            " at the glass wool production hall?\n"
            "2. Do you observe time trends or sudden shifts? Describe them with approximate times.\n"
            "3. Is there any indication of black carbon, brown carbon, mineral particles, or other types?\n"

            f"DATA (CSV format):\n\n{csv_text}\n\n"
        ),
    },
]

# Check the number of tokens used in the prompt. 
# The gpt-oss 20B model has a max token number of 131,072, but this includes both input and output.
# Therefore, max_input ≈ 131,072 − max_output. We should limit the output to half the total.

# Converts messages to a single string
def messages_to_text(messages):
    return "".join(
        f"{m['role']}:\n{m['content']}\n\n"
        for m in messages
    )

prompt_text = messages_to_text(full_input)

enc = tiktoken.get_encoding("cl100k_base")
num_tokens = len(enc.encode(prompt_text))
print("Estimated tokens:", num_tokens)

response = client.chat(
    model="gpt-oss",
    messages=full_input,
)

print(response["message"]["content"])

#%%
################################## Test 2 - Image interpretation #########################################

from pathlib import Path

img_bytes = Path(r"c:\Users\B279683\Desktop\NSOPS_NF_timeseries.png").read_bytes()

# Need to use qwen3 as gpt-oss cannot process images
response = client.chat(
    model="qwen3-vl:latest",
    messages=[{
        "role": "user",
        "content": "Describe the image.",
        "images": [img_bytes],
    }],
)

print(response["message"]["content"])

#%%
################################## Test 3 - Web access - own tool #########################################
# This test does not use a while loop, so the model is restricted to one online tool access
import requests

def internet_probe(url: str) -> str:
    """Fetch a URL and return its first 20000 characters."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.text[:20000]

messages: List[Dict[str, Any]] = [
    {"role": "user", "content": (
        "Use the internet_probe tool to access this url:\n"
        "https://httpbin.io/html/schema\n"
        "Summarize the content."
    )}
]

tools = [internet_probe]
available = {"internet_probe": internet_probe}

# 1) call the model with the initial prompt. The model assess whether it can answer
# directly or whether it needs to call the tool to get more information.
# No creative thinking is allowed (temperture:0)
resp = client.chat(
    model="gpt-oss", 
    messages=messages, 
    tools=tools, 
    options={"temperature": 0})

# 2) Extract the output of the model and check if it asked to use the tool
messages.append(resp["message"])
tool_calls = (resp["message"].get("tool_calls"))

# 3) Execute tool calls as ordered by the model and generate a new message based on tool output
for call in tool_calls:
    fn_name = call["function"]["name"]
    args = call["function"]["arguments"]

    result = available[fn_name](**args)

    messages.append({
        "role": "tool",
        "tool_name": fn_name,
        "content": result
    })

# 4) Make a new prompt to the model with the tool output
resp2 = client.chat(
    model="gpt-oss", 
    messages=messages, 
    tools=tools, 
    options={"temperature": 0})

# 5) print the logical steps (thinking) performed by the model
print("Thinking Process: ")
pprint(resp2["message"]["thinking"],width=70)
print()

# 6) print the content as returned from the model. 
print("Final Response: ")
print(resp2["message"]["content"])

#%%
##################### Test 4 - Web access - using ollama tools ############################
# This requires an API key, but it can be generated for free on the ollama website enabling 
# use of web_search and web_fetch functions available through ollama.
# --- BE AWARE! --- 
# There is a limited number of we searches allowed via the ollama tools unless you 
# subscribe :(

# Tools that we expose to the model
available_tools = {'web_search': web_search, 'web_fetch': web_fetch}

# Allowlisted args per tool (prevents "unexpected keyword argument" errors)
ALLOWED_ARGS = {
    "web_search": {"query", "max_results"},
    "web_fetch": {"url"},
}

# Original message that we prompt with
messages: List[Dict[str, Any]] = [{
    'role': 'user',
    'content': "What is the newest research published by the National Research Centre for the Working Environment in Denmark?"
}]

# We use a while loop, to allow the model to call tools,
# the local machine to execute them and feed the response back to the model
while True:
    # Initial message that we prompt with
    response = client.chat(
        model='gpt-oss',
        messages=messages,
        tools=[web_search, web_fetch],
        think=True
    )

    # print what the model is thinking to follow its progress
    if getattr(response.message, "thinking", None):
        print('Thinking: ', response.message.thinking)

    # Print the message it comes up with
    if getattr(response.message, "content", None):
        print('Content: ', response.message.content)

    # Append it to the current "messages" so there is a history for the model to understand
    msg = response.message
    messages.append(msg.model_dump())

    # If the model asks for a tool we print its request, to show what is happening
    if getattr(response.message, "tool_calls", None):
        print('Tool calls: ', response.message.tool_calls)

        # We go through all the tool calls that the model asked for
        for tool_call in response.message.tool_calls: # type:ignore
            tool_name = tool_call.function.name

            # We get the tool from the dict of available tools
            function_to_call = available_tools.get(tool_name)

            # if we found the desired tool:
            if function_to_call:
                # We get the arguments the model asked us to use
                args = dict(tool_call.function.arguments or {})

                # Filter args to only what the tool actually supports (avoid hallucinated kwargs like "source")
                allowed = ALLOWED_ARGS.get(tool_name, set())
                args = {k: v for k, v in args.items() if k in allowed}

                # We feed the arguments to the relevant function
                result = function_to_call(**args)

                # We print a snippet of the results
                print('Result: ', str(result)[:200] + '...')

                # Result is truncated for limited context lengths - so all our tokens aren't blown!
                messages.append({
                    'role': 'tool',
                    'content': str(result)[:2000 * 4],
                    'tool_name': tool_name
                })
            else:
                # If we did not find the relevant tool, we let the model know
                messages.append({
                    'role': 'tool',
                    'content': f'Tool {tool_name} not found',
                    'tool_name': tool_name
                })
    else:
        # If no tools were called, we can end the chat
        break

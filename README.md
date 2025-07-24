🧠 GenAI App using Amazon Bedrock + Streamlit
This is a simple GenAI (Generative AI) application built using Amazon Bedrock and Streamlit. It allows users to input prompts and receive responses from foundational models (FMs) hosted on Bedrock, such as Anthropic Claude, Mistral, or Amazon Titan.

🚀 Features
✅ Streamlit-based interactive UI

✅ Prompt-based interaction with Bedrock foundation models

✅ Supports text generation

✅ Easy-to-deploy on local or cloud environments

📦 Requirements
Make sure you have the following:

Python 3.8+

AWS Account with Bedrock access

IAM user/role with permission: bedrock:InvokeModel

Installed packages from requirements.txt

🔐 AWS Setup
Configure your AWS credentials:

bash
Copy
Edit
aws configure
Or export them directly:

bash
Copy
Edit
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=us-east-1
Ensure IAM permissions:

json
Copy
Edit
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel"
  ],
  "Resource": "*"
}
📁 Project Structure
pgsql
Copy
Edit
genai-app/
├── app.py                # Main Streamlit app
├── bedrock_client.py     # Wrapper for Bedrock API call
├── requirements.txt      # Python dependencies
└── README.md             # This file
🔧 Installation
Clone the repo:

bash
Copy
Edit
git clone https://github.com/your-username/genai-bedrock-streamlit.git
cd genai-bedrock-streamlit
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
Then open your browser at: http://localhost:8501

🧪 Sample Code
app.py
python
Copy
Edit
import streamlit as st
from bedrock_client import generate_response

st.title("🧠 GenAI with Bedrock")

prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        response = generate_response(prompt)
        st.write("### Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt.")
bedrock_client.py
python
Copy
Edit
import boto3
import json

def generate_response(prompt):
    bedrock = boto3.client('bedrock-runtime', region_name="us-east-1")

    body = json.dumps({"prompt": prompt, "max_tokens_to_sample": 200})
    response = bedrock.invoke_model(
        modelId="anthropic.claude-v2",  # Replace as needed
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response['body'].read())
    return result.get("completion", "No response.")
📘 Notes
Ensure your AWS user has Bedrock service access and quota.

Bedrock models are region-specific; ensure you're using the correct one.

You can change model ID based on what's available in your AWS account.

📄 License
MIT License

import streamlit as st
import boto3
import json
from datetime import datetime
from typing import Dict, List, Optional
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Bedrock Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BedrockAgentClient:
    def __init__(self, region_name: str = "ap-southeast-1"):
        """Initialize Bedrock Agent client"""
        self.region_name = region_name
        self.bedrock_agent_runtime = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Bedrock Agent Runtime client"""
        try:
            self.bedrock_agent_runtime = boto3.client(
                'bedrock-agent-runtime',
                region_name=self.region_name
            )
        except Exception as e:
            st.error(f"Failed to initialize Bedrock client: {str(e)}")
    
    def invoke_agent(self, agent_id: str, agent_alias_id: str, session_id: str, 
                    input_text: str, enable_trace: bool = False) -> Dict:
        """Invoke the Bedrock agent with the given input"""
        try:
            response = self.bedrock_agent_runtime.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=input_text,
                enableTrace=enable_trace
            )
            return response
        except Exception as e:
            st.error(f"Error invoking agent: {str(e)}")
            return None
    
    def parse_agent_response(self, response) -> List[Dict]:
        """Parse the streaming response from Bedrock agent"""
        if not response:
            return []
        
        messages = []
        try:
            event_stream = response['completion']
            
            for event in event_stream:
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        # Decode the response
                        message = chunk['bytes'].decode('utf-8')
                        messages.append({
                            'type': 'text',
                            'content': message,
                            'timestamp': datetime.now()
                        })
                elif 'trace' in event:
                    # Handle trace information if needed
                    trace = event['trace']
                    messages.append({
                        'type': 'trace',
                        'content': trace,
                        'timestamp': datetime.now()
                    })
        except Exception as e:
            st.error(f"Error parsing response: {str(e)}")
        
        return messages

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if 'bedrock_client' not in st.session_state:
    st.session_state.bedrock_client = None

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # AWS Configuration
    st.subheader("AWS Settings")
    region = st.selectbox(
        "AWS Region",
        ["ap-southeast-1"],
        index=0
    )
    
    # Agent Configuration
    st.subheader("Bedrock Agent Settings")
    agent_id = st.text_input(
        "Agent ID",
        value="CBWZKLEVNN",
        placeholder="Enter your Bedrock Agent ID",
        help="Your Bedrock Agent's unique identifier"
    )
    
    agent_alias_id = st.text_input(
        "Agent Alias ID",
        value="TSTALIASID",
        help="Agent alias (default: TSTALIASID for test)"
    )
    
    enable_trace = st.checkbox(
        "Enable Tracing",
        value=False,
        help="Enable to see detailed agent execution traces"
    )
    
    # Session Management
    st.subheader("Session Management")
    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    
    if st.button("🔄 New Session"):
        st.session_state.session_id = f"session_{int(time.time())}"
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main interface
st.title("🤖 Bedrock Agent Chat Interface")
st.markdown("Interact with your Amazon Bedrock agent through this streamlined interface.")

# Initialize Bedrock client
if not st.session_state.bedrock_client:
    st.session_state.bedrock_client = BedrockAgentClient(region_name=region)

# Validate configuration
if not agent_id:
    st.warning("⚠️ Please enter your Bedrock Agent ID in the sidebar to get started.")
    st.info("""
    **To use this application:**
    1. Enter your Bedrock Agent ID in the sidebar
    2. Optionally modify the Agent Alias ID (default: TSTALIASID)
    3. Ensure your AWS credentials are configured
    4. Start chatting with your agent!
    """)
    st.stop()

# Chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "trace" and enable_trace:
                with st.expander("🔍 Agent Trace"):
                    st.json(message["content"])
            else:
                st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your Bedrock agent anything..."):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is thinking..."):
            response = st.session_state.bedrock_client.invoke_agent(
                agent_id=agent_id,
                agent_alias_id=agent_alias_id,
                session_id=st.session_state.session_id,
                input_text=prompt,
                enable_trace=enable_trace
            )
            
            if response:
                messages = st.session_state.bedrock_client.parse_agent_response(response)
                
                # Combine text messages
                full_response = ""
                trace_data = None
                
                for msg in messages:
                    if msg['type'] == 'text':
                        full_response += msg['content']
                    elif msg['type'] == 'trace':
                        trace_data = msg['content']
                
                if full_response:
                    st.markdown(full_response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now()
                    })
                    
                    # Add trace information if available and enabled
                    if trace_data and enable_trace:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": trace_data,
                            "type": "trace",
                            "timestamp": datetime.now()
                        })
                else:
                    error_msg = "No response received from the agent."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })
            else:
                error_msg = "Failed to get response from agent. Please check your configuration."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))

with col2:
    st.metric("Current Session", st.session_state.session_id[-8:])

with col3:
    if st.session_state.messages:
        last_message_time = st.session_state.messages[-1]["timestamp"]
        st.metric("Last Activity", last_message_time.strftime("%H:%M:%S"))

# Help section
with st.expander("ℹ️ Help & Setup"):
    st.markdown("""
    ### Prerequisites
    1. **AWS Credentials**: Ensure your AWS credentials are configured via:
       - AWS CLI (`aws configure`)
       - Environment variables
       - IAM roles (if running on EC2)
    
    2. **Bedrock Agent**: You need a deployed Bedrock agent with:
       - Agent ID
       - Agent Alias (default test alias: TSTALIASID)
    
    3. **Required Permissions**:
       - `bedrock:InvokeAgent`
       - Access to your specific agent resource
    
    ### Installation
    ```bash
    pip install streamlit boto3
    ```
    
    ### Running the App
    ```bash
    streamlit run app.py
    ```
    
    ### Features
    - **Real-time Chat**: Interactive conversation with your Bedrock agent
    - **Session Management**: Maintain conversation context
    - **Trace Debugging**: View detailed agent execution traces
    - **Multi-region Support**: Configure different AWS regions
    - **Response Streaming**: Handle streaming responses from agents
    """)
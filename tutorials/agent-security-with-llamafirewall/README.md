![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-security-with-llamafirewall--readme)

# LlamaFirewall For AI Agents Security Tutorial

A comprehensive tutorial demonstrating how to implement security measures for AI agents using LlamaFirewall.

## Authors
This tutorial was created by [Matan Kotick](https://www.linkedin.com/in/matan-kotick-664735252) and [Amit Ziv](https://www.linkedin.com/in/amit-ziv-49690b120).

## Understanding Agent Security Risks

AI agents face significant security challenges across three main attack surfaces:

### Input Security
- **Prompt Injection**: Attempts to override agent instructions
- **Harmful Content**: Detection of inappropriate or dangerous inputs
- **Context Manipulation**: Prevention of memory/context tampering

### Output Security
- **Behavior Alignment**: Ensuring responses match intended purpose
- **Content Safety**: Preventing harmful or misleading outputs
- **Information Control**: Managing sensitive data disclosure

### Tool Security
- **Access Control**: Preventing unauthorized tool usage
- **Usage Monitoring**: Detecting malicious tool manipulation
- **Resource Protection**: Preventing system abuse

These vulnerabilities can lead to security breaches, misuse of resources, and potential harm to users or systems. This tutorial demonstrates how to address these challenges using LlamaFirewall.

**Note**: Additional attack surfaces may exist depending on the agent's implementation and deployment context. For example, supply chain risks around model sources, training data, and third-party tools must be considered. Infrastructure security for model hosting and API endpoints, as well as compliance requirements for audit logging and policy enforcement, are also critical concerns that should be addressed as part of a comprehensive security strategy.



## About This Tutorial

This tutorial demonstrates how to implement security measures using LlamaFirewall, an open-source security framework. While the examples are not production-ready code, they illustrate key security concepts that can be adapted for real-world applications.

### What is LlamaFirewall?

LlamaFirewall is a comprehensive open-source security framework designed to protect AI agents from various threats. It provides three main guardrails:

1. **PromptGuard 2**: Protects against prompt injection attacks
2. **Agent Alignment Checks**: Ensures agent behavior aligns with intended instructions
3. **CodeShield**: Prevents generation of harmful or malicious code

## Tutorials

This repository contains several Jupyter notebooks demonstrating how to implement security concepts for AI agents using LlamaFirewall:

- Input validation to protect against malicious prompts and harmful content
- Output validation to ensure agent responses align with intended behavior
- Tool calling security to prevent unauthorized or dangerous actions

While these tutorials use the OpenAI Agents SDK, the security concepts can be applied to any AI agent platform.

1. [Hello Llama](hello-llama.ipynb) - Demonstrates basic message scanning to detect and block potentially harmful content.
2. [Input Guardrail](input-guardrail.ipynb) - Shows how to validate user inputs to protect against malicious prompts.
3. [Output Guardrail](output-guardrail.ipynb) - Illustrates how to validate AI agent responses to ensure they align with intended behavior and don't contain harmful content
4. [Tools Security](tools-security.ipynb) - Covers comprehensive security for AI agent tools, including input validation, output scanning, and protection against external tool usage

## Prerequisites

### System Requirements
- Python 3.9+
- OpenAI API key
- Required packages (see requirements.txt)
- ~2GB of disk space for model downloads

### Model Requirements
LlamaFirewall uses HuggingFace models that require access permissions:
- Models are downloaded to `~/.cache/huggingface` by default
- Models are cached locally after first download
- **Important**: The Llama Prompt Guard 2 model requires explicit access permission

### API Keys
- **OpenAI API key** (required for the agent)
- **Together API key** (required for Alignment Checker scanner)
  - Get it from [Together AI](https://www.together.ai)
  - Required for `output_guardrail`

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TOGETHER_API_KEY=your_together_api_key_here    # Required for Alignment Checker scanner
   ```

4. Set up HuggingFace access:
   1. Create a HuggingFace account
   2. Request access to [Llama Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M):
      - Click "Access repository"
      - Provide your legal name, DOB, and organization
      - Accept the Llama 4 Community License
   3. Create a read access token in Settings
   4. Enter token when prompted on first run

5. Run `llamafirewall configure` for:
    1. Check if required models are available locally
    2. Help you download models from HuggingFace if they are not available
    3. Check if your environment has the required API key for certain scanners 

**Note:** This tutorial uses HuggingFace models, which will be downloaded to `~/.cache/huggingface`.

## Monitoring

This tutorial uses the OpenAI Agents SDK, which provides comprehensive monitoring and tracing capabilities through the OpenAI dashboard. All agent activities, including guardrail invocations, are automatically logged and can be monitored in real-time.

### Dashboard Access
- Log into your [OpenAI account dashboard](https://platform.openai.com/)
- Navigate to "Dashboard" section
- View detailed traces of agent interactions and guardrail checks

![OpenAI Dashboard showing LlamaFirewall guardrail invocation](assets/openai-trace.png)

This monitoring capability helps ensure transparency and allows you to verify that the guardrails are working as intended.


### Additional Resources
* [LlamaFirewall: An open source guardrail system for building secure AI agents](https://arxiv.org/pdf/2505.03574)
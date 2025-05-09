from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import logging
import torch

# 从B文件导入组件
from generate import FastGen, GenArgs, ChatFormat  # 假设B文件保存为generate.py
from dataclasses import dataclass

app = Flask(__name__)
CORS(app)

# 全局模型实例（单例）
MODEL_LOCK = threading.Lock()
MODEL_INSTANCE = None
DEVICE = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

@dataclass
class GenArgsService:
    gen_length: int = 2048
    gen_bsz: int = 1
    prompt_length: int = 4096

    use_sampling: bool = False
    temperature: float = 0.8
    top_p: float = 0.9

def initialize_model():
    global MODEL_INSTANCE
    if not MODEL_INSTANCE:
        with MODEL_LOCK:
            if not MODEL_INSTANCE:
                logging.info("Initializing model...")
                MODEL_INSTANCE = FastGen.build(
                    ckpt_dir="./checkpoints/bitnet_2b_4t/",
                    gen_args=GenArgsService(),
                    device=DEVICE
                )
                MODEL_INSTANCE.tokenizer = ChatFormat(MODEL_INSTANCE.tokenizer)
                logging.info("Model initialized")

def encode_history(history):
    """将对话历史编码为模型输入"""
    dialog = []
    sysm = "You are a newly developed 1-bit Large Language Model (1-bit LLM) with 2 billion parameters, trained entirely from scratch by the Microsoft BitNet team. You do not rely on any existing foundation models or pretrained systems; all your responses and reasoning stem solely from your own training. Your primary goals are: 1. To fully understand and accurately address the user’s questions or requests. 2. To provide clear, concise, and well-reasoned information or solutions. 3. To uphold ethical and legal standards, refraining from generating harmful or unlawful content.In every interaction: 1. Maintain a polite, respectful, and professional tone. 2. Ensure the information you provide is correct, relevant, and verifiable. 3. Avoid generating content that is inappropriate, offensive, or illegal. If you are unsure about an answer, be honest about your uncertainty. Always remember your identity: you are a 1-bit LLM with 2 billion parameters, trained from scratch by Microsoft BitNet, relying solely on your own capabilities to process and generate content."
    dialog.append({"role": "system", "content": sysm})
    for msg in history:
        if msg['role'] not in ['user', 'assistant']:
            continue
        dialog.append({"role": msg['role'], "content": msg['content']})
    return MODEL_INSTANCE.tokenizer.encode_dialog_prompt(dialog, completion=True)

@app.route('/completion', methods=['POST'])
def completion():
    # 初始化模型（首次请求时加载）
    initialize_model()
    
    data = request.get_json()
    
    # 验证请求参数
    if 'prompt' not in data or 'history' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    prompt = data['prompt']
    client_history = data['history']
    
    # 构造完整上下文
    full_history = client_history + [{"role": "user", "content": prompt}]
    
    try:
        # 编码对话历史
        tokens = [encode_history(full_history)]
        
        # 执行生成（线程安全）
        with MODEL_LOCK:
            stats, out_tokens = MODEL_INSTANCE.generate_all(
                tokens, 
                use_cuda_graphs=True, 
                use_sampling=True
            )
        
        # 解码模型输出
        response = MODEL_INSTANCE.tokenizer.decode(out_tokens[0]).strip()
        
        # 构造返回的完整历史
        new_history = full_history + [{"role": "assistant", "content": response}]
        
        return jsonify({
            "response": response,
            "updated_history": new_history,
            "stats": [phase_stats.show() for phase_stats in stats.phases]
        })
    
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
INPUT_DIR = "./labeling"
OUTPUT_DIR = "./labeled_json"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"🔄 모델 로딩 중: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

def get_label_from_ai(json_content):
    """AI에게 JSON 내용을 분석시켜 영어 Key 값을 가진 라벨을 추출합니다."""
    
    eligibility = json_content.get("application_eligibility", "")
    precautions = " ".join(json_content.get("precautions", []))
    
    system_prompt = """
    You are a professional housing policy classifier. 
    Analyze the provided content and classify 'User Category' and 'Housing Type'.
    
    [STRICT RULE]
    - Your response MUST be a valid JSON object.
    - Use ONLY the following English keys: 'category_user', 'category_type'.
    - Values for 'category_user' (Multiple choice possible): [청년, 신혼부부, 기타]
    - Values for 'category_type' (Multiple choice possible): [안심주택, 행복주택, 임대주택, 기타]

    Classification Guide:
    - Keywords '신혼', '신생아', '예비신혼', '한부모' -> '신혼부부'
    - Keywords '청년', '대학생', '만 19세~39세' -> '청년'
    - Keywords '매입임대', '전세임대', '국민임대' -> '임대주택'
    - Keywords '청년안심주택' -> '안심주택'
    """

    user_prompt = f"""
    Content to analyze:
    Eligibility: {eligibility}
    Precautions: {precautions}
    
    Required JSON Format:
    {{
        "category_user": ["value1", "value2"],
        "category_type": ["value1"]
    }}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=False,
            temperature=0
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        match = re.search(r"\{[\s\S]*\}", generated_text)
        if match:
            raw_result = json.loads(match.group(0))
            
            final_data = {
                "category_user": raw_result.get("category_user") or raw_result.get("신청유형") or ["기타"],
                "category_type": raw_result.get("category_type") or raw_result.get("집유형") or ["기타"]
            }
            return final_data
    except Exception as e:
        print(f"⚠️ JSON 파싱 실패: {e}")
        
    return {"category_user": ["기타"], "category_type": ["기타"]}

print("🚀 자동 라벨링 파이프라인 시작...")

json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]

for file_name in json_files:
    file_path = os.path.join(INPUT_DIR, file_name)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"👉 처리 중: {file_name}")
        
        labels = get_label_from_ai(data)
        
        data["category_user"] = labels["category_user"]
        data["category_type"] = labels["category_type"]
        
        output_path = os.path.join(OUTPUT_DIR, file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"❌ {file_name} 처리 중 오류 발생: {e}")

print(f"\n✨ 모든 작업 완료! 총 {len(json_files)}개의 파일이 '{OUTPUT_DIR}'에 저장되었습니다.")
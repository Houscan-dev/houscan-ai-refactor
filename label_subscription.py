import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
INPUT_DIR = "./extracted_json"
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
    title = json_content.get("title", "")
    pdf_name = json_content.get("pdf_name", "")
    
    system_prompt = """
    You are a professional housing policy classifier. 
    Analyze the provided content (including title and filename) and classify 'User Category' and 'Housing Type'.
    
    [STRICT RULES]
    - Your response MUST be a valid JSON object.
    - Use ONLY the following English keys: 'category_user', 'category_type'.
    - 'category_user' can have MULTIPLE values (array): [청년, 신혼부부, 기타]
    - 'category_type' should have ONLY ONE value: [안심주택, 행복주택, 임대주택, 기타]

    Classification Priority (Use title/filename FIRST):
    
    **User Category Decision Rules (MULTIPLE SELECTION ALLOWED):**
    - Check if '청년' keywords exist (청년, 대학생, 만 19세~39세) -> Include '청년'
    - Check if '신혼부부' keywords exist (신혼, 신생아, 예비신혼, 미리내집, 한부모) -> Include '신혼부부'
    - If BOTH '청년' AND '신혼부부' keywords found -> Return ["청년", "신혼부부"]
    - If ONLY '청년' keywords found -> Return ["청년"]
    - If ONLY '신혼부부' keywords found -> Return ["신혼부부"]
    - If NEITHER found -> Return ["기타"]
    - IMPORTANT: '기타' should NEVER be mixed with other categories. Use '기타' ONLY when neither 청년 nor 신혼부부 applies.
    
    **Housing Type Decision Tree (SINGLE SELECTION ONLY):**
    1. If title/filename contains '청년안심주택', '안심주택', '역세권청년주택' -> '안심주택' (FINAL)
    2. If title/filename contains '행복주택' -> '행복주택' (FINAL)
    3. If title/filename contains '매입임대', '전세임대' -> '임대주택' (FINAL)
    4. Otherwise, check content:
       - '청년안심주택', '역세권' keywords -> '안심주택'
       - '행복주택' keyword -> '행복주택'
       - '매입임대', '전세임대', '국민임대' -> '임대주택'
       - Otherwise -> '기타'
    
    **Important:** 
    - category_user: Can have multiple values (청년, 신혼부부), but exclude '기타' if other categories apply
    - category_type: Choose ONLY ONE value
    - Title/filename keywords take absolute priority over content.
    """

    user_prompt = f"""
    Announcement to classify:
    
    Title: {title}
    PDF Filename: {pdf_name}
    Eligibility: {eligibility}
    Precautions: {precautions}
    
    Analyze the title and filename FIRST, then the content.
    
    For category_user: Return ALL applicable values (청년, 신혼부부, or both if both apply)
    For category_type: Return ONLY ONE value
    
    Required JSON Format:
    {{
        "category_user": ["value1", "value2"],
        "category_type": "single_value"
    }}
    
    Examples:
    - If both 청년 and 신혼부부 apply: {{"category_user": ["청년", "신혼부부"], "category_type": "안심주택"}}
    - If only 청년 applies: {{"category_user": ["청년"], "category_type": "임대주택"}}
    - If only 신혼부부 applies: {{"category_user": ["신혼부부"], "category_type": "행복주택"}}
    - If neither applies: {{"category_user": ["기타"], "category_type": "기타"}}
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
            
            # category_user: 리스트로 정규화 (복수 선택 가능)
            user_cat = raw_result.get("category_user", ["기타"])
            if not isinstance(user_cat, list):
                user_cat = [user_cat]
            
            # '기타'가 다른 카테고리와 함께 있으면 제거
            if len(user_cat) > 1 and "기타" in user_cat:
                user_cat = [cat for cat in user_cat if cat != "기타"]
            
            # category_type: 단일 값으로 정규화 (단일 선택)
            type_cat = raw_result.get("category_type", "기타")
            if isinstance(type_cat, list):
                type_cat = type_cat[0] if type_cat else "기타"
            
            final_data = {
                "category_user": user_cat,      # 배열 (복수 가능)
                "category_type": [type_cat]     # 배열 (단일 값만)
            }
            return final_data
    except Exception as e:
        print(f"⚠️ JSON 파싱 실패: {e}")
        print(f"   생성된 텍스트: {generated_text[:200]}")
        
    return {"category_user": ["기타"], "category_type": ["기타"]}

print("🚀 자동 라벨링 파이프라인 시작...")

json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]

for file_name in json_files:
    file_path = os.path.join(INPUT_DIR, file_name)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # title과 pdf_name이 없는 경우 경고
        if not data.get("title") or not data.get("pdf_name"):
            print(f"⚠️ {file_name}: title 또는 pdf_name이 없습니다. 정확도가 낮을 수 있습니다.")
        
        print(f"👉 처리 중: {file_name}")
        print(f"   제목: {data.get('title', 'N/A')}")
        
        labels = get_label_from_ai(data)
        
        data["category_user"] = labels["category_user"]
        data["category_type"] = labels["category_type"]
        
        print(f"   ✅ User: {labels['category_user']}, Type: {labels['category_type']}")
        
        output_path = os.path.join(OUTPUT_DIR, file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"❌ {file_name} 처리 중 오류 발생: {e}")

print(f"\n✨ 모든 작업 완료! 총 {len(json_files)}개의 파일이 '{OUTPUT_DIR}'에 저장되었습니다.")
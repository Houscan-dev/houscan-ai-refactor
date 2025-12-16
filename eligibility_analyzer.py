import json
from dotenv import load_dotenv
import os
from typing import Dict, Any, List, Optional
from groq import Groq
import re
from datetime import datetime, date
import glob 
import sys 

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("GROQ_API_KEY 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)
    
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct" 


def extract_json(text: str):
    """LLM 응답 텍스트에서 JSON 객체만을 추출합니다."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        raw_json = text[start:end]
        
        clean_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', raw_json)
        
        return clean_json
    except:
        return text
    
def parse_financial_limit_from_criteria(criteria_text: str, keyword: str) -> Optional[int]:
    """
    기준 텍스트에서 금액을 찾아 정수로 변환합니다. (예: '2억 9900만원' -> 299000000)
    """
    if keyword not in criteria_text:
        return None

    start_index = criteria_text.find(keyword)
    search_text = criteria_text[start_index:]
    
    pattern = re.compile(r'([\d,]+\s*억\s*[\d,]*\s*만?\s*원|[\d,]+\s*만\s*원|[\d,]+\s*원)')
    
    match = pattern.search(search_text)
    
    if not match:
        return None

    amount_str = match.group(0) 
    
    try:
        cleaned_str = amount_str.replace(',', '').replace('원', '').replace(' ', '')
        
        total_amount = 0
        
        if '억' in cleaned_str:
            parts = cleaned_str.split('억')
            if parts[0].isdigit():
                total_amount += int(parts[0]) * 100000000
            
            if '만' in parts[1]:
                만_parts = parts[1].split('만')
                if 만_parts[0].isdigit():
                    total_amount += int(만_parts[0]) * 10000 
            
            elif parts[1].isdigit():
                    total_amount += int(parts[1])

        elif '만' in cleaned_str:
            pure_number = cleaned_str.replace('만', '')
            if pure_number.isdigit():
                total_amount = int(pure_number) * 10000

        elif cleaned_str.isdigit():
                total_amount = int(cleaned_str)
        
        if total_amount > 0:
            return total_amount

    except Exception:
        return None
    
    return None

def calculate_age(birth_date_str: str, announcement_date_str: str) -> Optional[int]:
    """공고일 기준으로 만 나이를 계산합니다. (birth_date: YYMMDD)"""
    try:
        if len(birth_date_str) == 6:
            current_year = date.today().year
            birth_year_prefix = 19 if int(birth_date_str[:2]) > (current_year % 100) + 1 else 20
            birth_date_str = str(birth_year_prefix) + birth_date_str
            
        birth_date = datetime.strptime(birth_date_str, "%Y%m%d").date()
        
        announcement_date_str = announcement_date_str.rstrip('.') 
        announcement_date = datetime.strptime(announcement_date_str, "%Y.%m.%d").date()
        
        age = announcement_date.year - birth_date.year - ((announcement_date.month, announcement_date.day) < (birth_date.month, birth_date.day))
        return age
    except Exception:
        return None
    
def preprocess_user_data(user_data: Dict[str, Any], notice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 데이터를 공고문 기준과 비교하여 LLM에게 최종 판단 상태를 전달합니다.
    """
    processed_data = {}
    
    announcement_date_str = notice_data.get("application_schedule", {}).get("announcement_date", "2025.01.01")
    
    all_criteria_text = notice_data.get("application_eligibility", "")
    for p in notice_data.get("priority_and_bonus", {}).get("priority_criteria", []):
        all_criteria_text += " " + " ".join(p.get("criteria", []))
    
    user_age = calculate_age(user_data.get("birth_date", ""), announcement_date_str)
    
    NOTICE_AGE_MIN = 19
    NOTICE_AGE_MAX = 39 
    
    processed_data["user_age"] = user_age
    
    if user_age is None:
        processed_data["age_status"] = "⚠️ 판단 불가: 생년월일 형식 오류 또는 공고일 미기재"
    elif user_age < NOTICE_AGE_MIN:
        processed_data["age_status"] = f"❌ 나이 기준 미달 (만 {user_age}세 < 만 {NOTICE_AGE_MIN}세)"
    elif user_age > NOTICE_AGE_MAX:
        processed_data["age_status"] = f"❌ 나이 기준 초과 (만 {user_age}세 > 만 {NOTICE_AGE_MAX}세)"
    else:
        processed_data["age_status"] = "🟢 나이 기준 충족" 
    
    notice_asset_max = parse_financial_limit_from_criteria(all_criteria_text, "자산")
    notice_car_max = parse_financial_limit_from_criteria(all_criteria_text, " 자동차")

    user_total_assets = user_data.get("total_assets", 0)
    user_car_value = user_data.get("car_value", 0)

    processed_data["asset_status"] = "🟢 자산 기준 충족"
    if notice_asset_max is None:
        processed_data["asset_status"] = "⚠️ 판단 불가: 공고문 기준 자산액 추출 실패"
    elif user_total_assets > notice_asset_max:
        processed_data["asset_status"] = f"❌ 총 자산 기준 초과 ({user_total_assets:,}원 > {notice_asset_max:,}원)"
    else:
        if notice_asset_max is not None:
            processed_data["asset_status"] = f"🟢 자산 기준 충족 ({user_total_assets:,}원 / {notice_asset_max:,}원)"

    processed_data["car_status"] = "🟢  자동차 가액 기준 충족"
    if notice_car_max is None:
        processed_data["car_status"] = "⚠️ 판단 불가:  자동차 가액 기준액 추출 실패"
    elif user_car_value > notice_car_max:
        processed_data["car_status"] = f"❌  자동차 가액 기준 초과 ({user_car_value:,}원 > {notice_car_max:,}원)"
    else:
        if notice_car_max is not None:
            processed_data["car_status"] = f"🟢  자동차 가액 기준 충족 ({user_car_value:,}원 / {notice_car_max:,}원)"
    
    return processed_data

def analyze_eligibility_with_ai(user_data: Dict[str, Any], notice_data: Dict[str, Any]) -> Dict[str, Any]:
    preprocessed_data = preprocess_user_data(user_data, notice_data) 
    client = Groq(api_key=GROQ_API_KEY)
    
    announcement_date_in_prompt = notice_data.get("application_schedule", {}).get("announcement_date", "정보 없음")
    priority_list = notice_data.get("priority_and_bonus", {}).get("priority_criteria", [])
    priority_text = "\n".join(
        [f"- {p.get('priority', '')}: {', '.join(p.get('criteria', []))}" for p in priority_list]
    )
    
    user_income_claim = user_data.get("income_range", "정보 없음") 
    
    user_welfare_status = '복지 수혜 대상자 (생계급여 수급자, 의료급여 수급자, 주거급여 수급자, 교육급여 수급자, 지원대상 한부모 가족, 차상위계층 가구에 해당)' \
                          if user_data.get("welfare_receipient", False) else '복지 수혜 대상자 아님'
                          
    field_description = """
### [LLM 판단용 입력 데이터 필드 정의]
- is_married: 결혼 여부 (true/false)
- welfare_receipient: 생계급여 수급자, 의료급여 수급자, 주거급여 수급자, 교육급여 수급자, 지원대상 한부모 가족, 차상위계층 가구 중 해당 여부 (true/false)
- parents_own_house: 부모님 주택 소유 여부 (true: 소유함, false: 소유하지 않음)
- disability_in_family: 본인 또는 가구원 중 장애인 등록증 소유 여부 (true/false)
- subscription_account: 청약 납입 횟수 (횟수만 전달됨)
- residence: 거주 자치구 (예: 성북구)
"""
    
    prompt_for_llm_first_pass = f"""
너는 청약 자격 검증 AI이다. **[2차 Python 검증 대상 항목]은 무시하고, [신청자격 요건], [우선순위 기준] 및 나머지 [사용자 정보]만을 기반으로 1차 결론을 도출해야 한다.**

{field_description}

출력 JSON 구조:
{{
  "is_eligible": true/false,
  "priority": "", # 예: "1순위"
  "reasons": [
        "최종 부적격 사유를 간결하고 정확하게 설명한 문장.",
        ...
    ]
}}

### 공고문 기준 정보
- 공고일: {announcement_date_in_prompt}
- 신청자격: {notice_data.get("application_eligibility", "정보 없음")}

### 사용자 정보 (1차 LLM 판단 근거)
- 복지 수혜 상태: {user_welfare_status}
- 주택 소유 상태: {'무주택' if not user_data.get('parents_own_house', True) else '부모님 주택 소유'}
- 사용자 소득 범위 주장: {user_income_claim} 
    # 소득 논리 강조
    # '50% 이하'는 '100% 이하'에 포함되는 개념이므로, 사용자가 50% 이하를 주장하면 100% 이하 조건은 충족함.
- 거주지: {user_data.get("residence", "정보 없음")}
- 결혼 상태: {'기혼' if user_data.get('is_married', False) else '미혼'}

### 2차 Python 검증 대상 항목 (LLM은 이 항목을 1차 판단에서 무시해야 함)
- 나이, 총자산,  자동차 가액 (이 항목들의 수치 판단은 Python이 최종적으로 강제합니다. LLM은 이와 관련된 사유를 reasons에 포함하지 마세요.)

### 우선순위 기준
{priority_text}

### [최고 우선순위 규칙]
1. **LLM의 1차 역할**: [2차 Python 검증 대상 항목]을 제외한 **모든 필수 비수치적 조건** (미혼 여부, 무주택 세대구성원 여부, 소득 기준 논리 등)을 검증하여 미충족 사유만 `reasons`에 생성합니다.
2. **ONLY FAILURE & NO WARNING**: `reasons`에는 오직 LLM이 찾은 **명확한 부적격 사유**만 포함해야 합니다. 충족 조건(🟢), 우대 사항, 나이/자산/ 자동차 관련 언급은 절대로 포함하지 마세요.
3. **출력 문장 통합 및 품질 유지**: 부적격 사유를 하나의 간결하고 포괄적인 문장으로 통합하고, **정확한 맞춤법과 띄어쓰기를 준수**해야 합니다.
4. **priority 결정**: 1차 판단 결과 `is_eligible`이 true일 경우에만 순위를 결정합니다.

**출력은 반드시 JSON만. 여분 설명 절대 금지.**
"""
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "너는 위에 제공된 정보만을 기반으로, 요청된 JSON 구조와 규칙을 완벽하게 준수하여 아무런 서문이나 설명 없이 순수 JSON 객체만을 출력하는 AI이다."},
                {"role": "user", "content": prompt_for_llm_first_pass}
            ],
            temperature=0.0,
            max_tokens=512
        )

        raw = completion.choices[0].message.content.strip()
        clean = extract_json(raw)
        ai_result = json.loads(clean)
        
    except Exception as e:
        return {
            "is_eligible": False,
            "priority": "",
            "reasons": ["미해당 사유 추출에 실패하였습니다. 잠시 후 다시 시도해 주세요."]
        }

    
    python_failure_reasons_sentences = []
    
    if "❌" in preprocessed_data['age_status']:
        user_age = preprocessed_data['user_age']
        python_failure_reasons_sentences.append(f"공고일({announcement_date_in_prompt}) 기준으로 만 {user_age}세이므로 나이 기준에 미달합니다.")
    
    if "❌" in preprocessed_data['asset_status']:
        asset_user = f"{user_data.get('total_assets', 0):,}"
        asset_limit = preprocessed_data['asset_status'].split(' / ')[1].rstrip(')')
        python_failure_reasons_sentences.append(f"총 자산이 기준액({asset_limit} 이하)을 초과하는 {asset_user}원입니다.")

    if "❌" in preprocessed_data['car_status']:
        car_user = f"{user_data.get('car_value', 0):,}"
        car_limit = preprocessed_data['car_status'].split(' / ')[1].rstrip(')')
        python_failure_reasons_sentences.append(f" 자동차 가액이 기준액({car_limit} 이하)을 초과하는 {car_user}원입니다.")

    llm_reasons = []
    for reason in ai_result.get("reasons", []):
        clean_reason = re.sub(r'\s+', ' ', reason).strip()
        
        is_warning_or_positive = any(phrase in clean_reason for phrase in [
            "⚠️", "판단 불가", "확인할 수 없", "검증할 수 없", "기준을 알 수 없", "정보가 부족", "충족합니다", "충족했습니다", "문제 없습니다"
        ])
        
        if clean_reason and not is_warning_or_positive:
            llm_reasons.append(clean_reason)
            
    final_reasons_raw = llm_reasons + python_failure_reasons_sentences
    
    final_is_eligible = bool(final_reasons_raw) == False
    final_priority = ai_result.get("priority", "")
    
    if final_is_eligible and not final_reasons_raw and not final_priority:
        final_priority = "3순위"

    final_reasons = []
    if final_reasons_raw:
          integration_prompt = f"다음은 신청자에게 해당되는 모든 부적격 사유 목록입니다. 이 사유들을 **하나의 간결하고 자연스러운 한국어 문장**으로 통합하여 출력해주세요. 다른 설명이나 서문은 포함하지 말고, 오직 문장만 출력해야 합니다. 사유 목록: {final_reasons_raw}"
          
          try:
              integration_completion = client.chat.completions.create(
                  model=GROQ_MODEL_NAME,
                  messages=[
                      {"role": "system", "content": "너는 주어진 사유 목록을 하나의 문장으로 통합하고, 정확한 맞춤법과 띄어쓰기를 준수하여 출력하는 AI이다. 다른 설명은 절대 추가하지 마시오."},
                      {"role": "user", "content": integration_prompt}
                  ],
                  temperature=0.1,
                  max_tokens=256
              )
              integrated_reason = integration_completion.choices[0].message.content.strip()
              final_reasons.append(re.sub(r'\s+', ' ', integrated_reason).strip())
          except Exception:
              final_reasons = final_reasons_raw

    return {
        "is_eligible": final_is_eligible,
        "priority": final_priority,
        "reasons": final_reasons
    }

if __name__ == "__main__":
    
    TEST_USER_DATA = {
        "age": 23,
        "birth_date": "030925",
        "gender": "F",
        "university": False,
        "graduate": True,
        "employed": False,
        "job_seeker": False,
        "is_married": False,
        "residence": "성북구",
        "welfare_receipient": True,
        "parents_own_house": False,
        "disability_in_family": False,
        "subscription_account": 0,
        "total_assets": 1000000,
        "car_value": 0,
        "income_range": "50% 이하",
    }

    json_files = glob.glob("./extracted_json/*.json")
    
    if not json_files:
        print("⚠️ 경고: './extracted_json/' 디렉터리에서 JSON 파일을 찾을 수 없습니다. 테스트를 진행할 수 없습니다.")
        sys.exit(0) 

    loaded_notices = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notice_data = json.load(f)
                loaded_notices.append((notice_data, os.path.basename(file_path)))
        except Exception as e:
            print(f"❌ 오류: 파일 {file_path} 로드 실패 - {e}")
            
    if not loaded_notices:
        print("⚠️ 경고: 로드에 성공한 공고 파일이 없습니다. 테스트를 진행할 수 없습니다.")
        sys.exit(0)

    for notice_data, filename in loaded_notices:
        print("\n" + "="*80)
        print(f"## 📋 공고 분석 시작: {filename} (ID: {notice_data.get('announcement_id', 'N/A')})")
        print("="*80)
        
        preprocessed = preprocess_user_data(TEST_USER_DATA, notice_data)
        print("--- 💡 처리된 사용자 상태 (Pre-processing 결과) ---")
        print(json.dumps(preprocessed, ensure_ascii=False, indent=2))
        print("-------------------------------------------------")
        
        result = analyze_eligibility_with_ai(TEST_USER_DATA, notice_data)

        print("\n--- ✅ 최종 AI 판단 결과 ---")
        print(f"is_eligible: {result['is_eligible']}")
        print(f"priority: \"{result['priority']}\"")
        print("reasons:", result["reasons"])
        print("====================================")
import os
import json
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Union
import torch
import re
import shutil

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.readers.file import PyMuPDFReader

from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb

class ApplicationSchedule(BaseModel):
    announcement_date: str | None = Field(None, description="모집공고일 (YYYY.MM.DD 형식 또는 null)")
    online_application_period: str | None = Field(
        None,
        description="신청 접수 기간. 시작일과 종료일이 모두 있으면 'YYYY.MM.DD~YYYY.MM.DD', 하루면 'YYYY.MM.DD'. 정보가 없으면 null."
    )
    document_submission_period: str | None = Field(
        None,
        description="서류 제출 기간. 시작과 끝이 모두 있으면 'YYYY.MM.DD~YYYY.MM.DD', 하루면 'YYYY.MM.DD'. 정보가 없으면 null."
    )
    document_announce_date: str | None = Field(None, description="서류심사 대상자 발표일 (YYYY.MM.DD 형식 또는 null)")
    winner_announcement: str | None = Field(None, description="당첨자(입주자) 발표일시 (YYYY.MM.DD HH:mm 또는 null)")
    contract_period: str | None = Field(
        None,
        description="계약 체결 기간. 시작과 끝이 모두 있으면 'YYYY.MM.DD~YYYY.MM.DD', 하루면 'YYYY.MM.DD'. 정보가 없으면 null."
    )
    move_in_period: str | None = Field(
        None,
        description="입주 기간. 시작과 끝이 모두 있으면 'YYYY.MM.DD~YYYY.MM.DD', 하루면 'YYYY.MM.DD'. 정보가 없으면 null."
    )

class PriorityCriteriaItem(BaseModel):
    priority: str = Field(..., description="순위 (예: 1순위, 2순위)")
    criteria: List[str] = Field(..., description="자격요건 설명 문자열 리스트")

class ScoreItem(BaseModel):
    item: str = Field(..., description="가점 항목명")
    score: str = Field(..., description="점수 (정수만, 다른 문자 없이 숫자만)")

class ScoreGroup(BaseModel):
    priority: str = Field(..., description="적용 대상 순위 (예: 1순위, 2순위, 공통)")
    items: List[ScoreItem] = Field(..., description="해당 대상에 적용되는 가점 항목 목록")

class PriorityAndBonus(BaseModel):
    priority_criteria: List[PriorityCriteriaItem] = Field(..., description="순위별 자격요건 목록. 명확히 제시되지 않으면 빈 배열([])을 반환하세요.")
    score_items: List[ScoreGroup] = Field(..., description="가점 항목 그룹 목록. 명확히 제시되지 않으면 빈 배열([])을 반환하세요.")

class DetailHouseInfo(BaseModel):
    name: str | None = Field(None, description="주택명. 정보가 없다면 null을 반환하세요.")
    address: str | None = Field(None, description="주소. 정보가 없다면 null을 반환하세요.")
    district: str | None = Field(None, description="자치구. 정보가 없다면 null을 반환하세요.")
    
    type: List[str] = Field(..., description="유형 (예: ['원룸형', '도시형생활주택']). 정보가 없다면 빈 배열([])을 반환하세요.")
    
    total_households: int | None = Field(None, description="총 세대수. 정보가 없다면 null을 반환하세요.")
    
    supply_households: List[int] = Field(..., description="공급호수 (단위 제거된 숫자 리스트). 이 리스트의 길이는 type, house_type과 동일해야 합니다.")
    
    house_type: List[str] = Field(..., description="주택형 (예: ['37A', '27B', '29m²']). 이 리스트의 길이는 type, supply_households와 동일해야 합니다.")
    
    parking: str | None = Field(None, description="주차장 정보 (예: 주차면수 15대). 정보가 없다면 null을 반환하세요.")
    
    elevator: bool | None = Field(None, description="엘리베이터 유무 (있음: true, 없음: false). 정보가 없다면 null을 반환하세요.")
    
    @field_validator('supply_households', mode='before')
    @classmethod
    def convert_supply_households(cls, v: Union[List[str], List[int], str]) -> List[int]:
        if isinstance(v, list):
            return [int(re.sub(r'[^0-9]', '', str(item))) if re.search(r'\d', str(item)) else 0 for item in v]
        elif isinstance(v, str):
            num_str = re.sub(r'[^0-9]', '', v)
            if num_str:
                return [int(num_str)]
        return v if isinstance(v, list) else []

class HousingNoticeSummary(BaseModel):
    announcement_id: int | None = Field(None, description="공고 식별자 (순차적으로 할당됨).")
    
    application_eligibility: str | None = Field(None, description="[신청 자격] 주요 연령, 소득, 자산 기준 요약.")
    housing_info: List[DetailHouseInfo] = Field(..., description="[공급 주택 정보] 공급 주택 정보 목록.")
    residence_period: str | None = Field(None, description="[거주 기간] 최대 기간과 갱신 조건.")
    priority_and_bonus: PriorityAndBonus = Field(..., description="[우선순위 및 가점사항] 구조화된 정보")
    application_schedule: ApplicationSchedule = Field(..., description="[모집 일정] 상세 모집 일정이 포함된 구조화된 정보")
    
    precautions: List[str] = Field(..., description="[유의사항] 가장 중요한 유의사항 2~3가지를 요약한 문자열 리스트.")

json_schema_str = HousingNoticeSummary.model_json_schema()

def clean_text(text: str) -> str:
    """PDF에서 추출된 텍스트의 품질을 높이기 위해 노이즈를 제거하고 정규화합니다."""
    text = re.sub(r'[●▩■◆◇※]', '', text)
    text = re.sub(r'[\s]{2,}', ' ', text)
    text = re.sub(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.', r'\1.\2.\3.', text)
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
    text = text.strip()
    return text

PERSIST_DIR = "./chroma_db"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "./pdfs")

db = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = db.get_or_create_collection("sh_public_housing_notices_hf")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

documents = []
reader = PyMuPDFReader()

try:
    chroma_collection.delete()
    print("Vector DB가 새로 로드될 PDF로 초기화되었습니다.")
except Exception as e:
    try:
        chroma_collection.delete(where={})
        print("Vector DB가 새로 로드될 PDF로 초기화되었습니다.")
    except Exception as e2:
        print(f"Vector DB 초기화 중 오류 발생: {e2}")

for file_name in os.listdir(DOCUMENTS_DIR):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(DOCUMENTS_DIR, file_name)

        try:
            doc_list = reader.load_data(file_path)
            
            for doc in doc_list:
                cleaned_text = clean_text(doc.text)
                metadata = doc.metadata.copy()
                metadata['file_name'] = file_name
                
                cleaned_node = TextNode(
                    text=cleaned_text,
                    metadata=metadata
                )
                
                documents.append(cleaned_node)

        except Exception as e:
            print(f"[{file_name}] 파일 로드 후 처리 중 오류 발생: {e}")

parser = SentenceSplitter(chunk_size=3072, chunk_overlap=300)
nodes = parser.get_nodes_from_documents(documents)

if not nodes:
    print("로드된 문서나 청크가 없습니다. PDF 경로와 파일 상태를 확인하세요.")
    exit()

pipeline = IngestionPipeline(
    transformations=[Settings.embed_model],
    vector_store=vector_store,
)
pipeline.run(nodes=nodes)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

print(f"총 {len(nodes)}개의 구조화된 청크가 Vector DB에 저장되었습니다.")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

def extract_json_object(text: str) -> str | None:
    match = re.search(r"\{[\s\S]*\}", text.strip())
    if not match:
        return None
    raw_json = match.group(0)
    raw_json = re.sub(r',\s*([\]}])', r'\1', raw_json)
    return raw_json

def normalize_housing_info(data: dict) -> dict:
    """
    LLM 응답 후처리를 통해 데이터 무결성을 높이고, 리스트 길이를 일치시킵니다.
    """
    if not isinstance(data, dict):
        return data
    if "housing_info" not in data or not isinstance(data["housing_info"], list):
        return data

    new_infos = []
    for h in data["housing_info"]:
        if not isinstance(h, dict):
            new_infos.append(h)
            continue

        district_val = h.get("district")
        temp_infos = []
        if isinstance(district_val, list):
            for d in district_val:
                new_h = h.copy()
                new_h["district"] = d
                temp_infos.append(new_h)
        else:
            temp_infos.append(h)
            
        for info in temp_infos:
            total_h = info.get("total_households")
            if isinstance(total_h, str):
                num_str = re.sub(r'[^0-9]', '', total_h)
                info["total_households"] = int(num_str) if num_str else None
            elif isinstance(total_h, (float, int)):
                info["total_households"] = int(total_h)

            supply_h = info.get("supply_households")
            if isinstance(supply_h, str):
                supply_h = [supply_h]
            if isinstance(supply_h, list):
                info["supply_households"] = supply_h
            else:
                 info["supply_households"] = []

            for key in ["type", "house_type"]:
                val = info.get(key)
                if isinstance(val, str):
                    if ',' in val:
                         info[key] = [item.strip() for item in val.split(',') if item.strip()]
                    else:
                        info[key] = [val]
                elif val is None:
                    info[key] = []
                elif not isinstance(val, list):
                    info[key] = [str(val)]

            type_list = info.get("type", [])
            supply_list = info.get("supply_households", [])
            house_type_list = info.get("house_type", [])

            target_len = max(len(supply_list), len(house_type_list))

            if target_len > 0:
                if len(type_list) == 1 and len(type_list) < target_len:
                    common_type = type_list[0]
                    info["type"] = [common_type] * target_len
                    type_list = info["type"]
                
                if len(type_list) != target_len or len(supply_list) != target_len or len(house_type_list) != target_len:
                    print(f"⚠️ 경고: 주택 정보 리스트 길이 불일치 (Type:{len(type_list)}, Supply:{len(supply_list)}, House:{len(house_type_list)}) - 파일: {data.get('file_name') or 'N/A'}")

            elevator_val = info.get("elevator")
            if isinstance(elevator_val, str):
                if '있' in elevator_val or '가능' in elevator_val or 'O' in elevator_val:
                    info["elevator"] = True
                elif '없' in elevator_val or '불가능' in elevator_val or 'X' in elevator_val:
                    info["elevator"] = False
                else:
                    info["elevator"] = None
            elif isinstance(elevator_val, (bool, int, float)):
                info["elevator"] = bool(elevator_val) if elevator_val in (True, 1, 1.0) else (False if elevator_val in (False, 0, 0.0) else None)
            else:
                 info["elevator"] = None
        
        new_infos.extend(temp_infos)

    if 'notice_id' in data:
        data.pop('notice_id')
        
    data["housing_info"] = new_infos
    
    precautions_val = data.get("precautions")
    if isinstance(precautions_val, str):
        if '.' in precautions_val and len(precautions_val) > 50:
             data["precautions"] = [p.strip() + '.' for p in precautions_val.split('.') if p.strip()]
        else:
             data["precautions"] = [precautions_val]
    elif precautions_val is None:
        data["precautions"] = []
    
    return data

def extract_structured_json_with_oss(file_name: str, index: VectorStoreIndex):
    """특정 공고문 파일명에 해당하는 내용을 검색하여 LLM JSON Mode로 JSON을 추출합니다."""
    
    queries_by_section = {
        "application_schedule": "모집 공고일, 신청 접수 기간, 서류 제출, 서류심사 대상자 발표일, 당첨자 발표, 계약 체결 등 전체 모집 일정 표의 내용",
        "supply_house_info": (
            "공급 주택 세부정보 표 전체. 주택명, 주소, 자치구, 유형, 총 세대수, 공급호수, 주택형(예: 37A, 29m²), 설치시설, 부대시설 "
            "주차장 정보, 엘리베이터 유무가 포함된 모든 행을 빠짐없이 추출"
        ),
        "priority_and_bonus": "입주 순위(1순위, 2순위) 자격 요건 표 및 가점 항목과 배점 표의 내용",
        "application_eligibility": "신청 자격 조건 (소득, 자산, 나이, 무주택 여부 등) 상세 설명",
        "residence_period": "최초 및 최대 거주 기간과 갱신 계약 조건",
        "precautions": "신청 시 반드시 알아야 할 가장 중요한 유의사항 목록"
    }

    retriever = index.as_retriever(
        filters=MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=file_name)]),
        similarity_top_k=12
    )

    all_retrieved_nodes = []
    unique_node_ids = set()

    print(f"[{file_name}] 타겟 검색 시작...")
    for section, query in queries_by_section.items():
        retrieved_nodes = retriever.retrieve(query)
        for node in retrieved_nodes:
            if node.node_id not in unique_node_ids:
                all_retrieved_nodes.append(node)
                unique_node_ids.add(node.node_id)

    if not all_retrieved_nodes:
        return {"error": "해당 공고문 파일에 대한 내용을 찾을 수 없습니다."}

    context_text = "\n\n--- 청약 공고문 내용 ---\n\n".join([n.get_content() for n in all_retrieved_nodes])
    print(f"  -> 총 {len(all_retrieved_nodes)}개의 고유한 청크로 최종 컨텍스트 구성 완료.")

    system_prompt = (
        "당신은 SH 청약 공고문 전문가입니다. 요청된 정보를 다음 JSON 스키마 형식에 맞춰 "
        "명확하고 간결한 한국어로 추출해야 합니다. **JSON 스키마에 정의된 모든 필드는 반드시 포함되어야 합니다.** "
        "\n\n"
        "### [주택 정보 추출 규칙 강화] ###\n"
        "1. 공급 주택 정보(housing_info)는 공고문에 나온 주택 수만큼 반드시 리스트로 모두 채워야 합니다.\n"
        "2. 각 주택 객체에는 name, address, district, parking을 문자열로, total_households를 정수(int)로 채워 넣으세요.\n"
        "3. **type, supply_households, house_type**은 해당 주택에 공급되는 모든 유형을 담는 **리스트(`[]`)** 로 추출하세요.\n"
        "   - **주의: type, supply_households, house_type 이 세 리스트의 길이는 반드시 동일해야 합니다.** (각 인덱스가 하나의 주택형 상세 정보를 나타내는 세트입니다.)\n"
        "   - **supply_households**의 값은 단위(세대, 호)를 제거한 **정수(int)** 만 포함해야 합니다. (예: [1, 1, 1])\n"
        "4. **elevator**는 '있음/가능'이면 `true`, '없음/불가능'이면 `false`, 정보가 불분명하면 `null`로 추출하세요.\n"
        "5. 만약 공고문에 '강동구, 광진구, 금천구'처럼 여러 자치구가 한 행에 표기되어 있다면, "
        "각 자치구별로 housing_info 객체를 각각 생성하세요. (즉, housing_info 배열의 길이를 늘리세요.)\n"
        "### [유의사항 추출 규칙 강화] ###\n"
        "1. **precautions** 필드는 가장 중요한 유의사항 2~3가지를 요약하여 **문자열 리스트(`List[str]`)** 로 추출하세요. 정보가 없으면 빈 배열([])을 반환하세요.\n"
        "\n"
        "**JSON 출력 형식**: 출력은 유효한 JSON 객체만 포함해야 합니다. 설명이나 마크다운(```json) 같은 것은 금지."
    )

    user_prompt = f"""
    아래 청약 공고문 내용을 분석하여, 다음 JSON 스키마에 맞춰 정보를 추출 및 요약해주세요.

    --- JSON SCHEMA ---
    {json_schema_str}

    ---
    청약 공고문 원본 내용:
    {context_text}
    ---
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        raw_json_output = extract_json_object(generated_text)
        if not raw_json_output:
            raise ValueError(f"LLM 응답에서 유효한 JSON 객체를 찾을 수 없습니다. 응답 일부: {generated_text[:400]}")

        try:
            parsed_dict = json.loads(raw_json_output)
        except json.JSONDecodeError as e:
            print("원본 LLM 응답(디버그):", generated_text)
            raise ValueError(f"LLM에서 추출한 JSON 파싱 실패: {e}")

        parsed_dict = normalize_housing_info(parsed_dict)

        try:
            pydantic_instance = HousingNoticeSummary.model_validate(parsed_dict)
            return pydantic_instance.model_dump()
        except ValidationError as ve:
            print("Pydantic 검증 실패. LLM 출력(원문 일부):", generated_text[:800])
            print("파싱된 JSON (사전):", json.dumps(parsed_dict, ensure_ascii=False, indent=2)[:2000])
            raise ve

    except Exception as e:
        print(f"JSON 추출 중 오류 발생 (파일: {file_name}): {e}")
        return {"error": f"JSON 생성 및 유효성 검사 실패: {e}"}

OUTPUT_DIR = "./extracted_json"
ERROR_DIR = "./error_pdfs"
EXTRACTED_PDF_DIR = "./extracted_pdfs"
SOURCE_PDF_DIR = "./pdfs"

for d in [OUTPUT_DIR, ERROR_DIR, EXTRACTED_PDF_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def get_next_announcement_id(extracted_dir: str) -> int:
    """extracted_dir에 있는 JSON 파일의 개수를 세어 다음 ID를 반환합니다."""
    count = 0
    if os.path.exists(extracted_dir):
        for name in os.listdir(OUTPUT_DIR): 
            if name.endswith('.json'):
                count += 1
    return count + 1

next_id = get_next_announcement_id(OUTPUT_DIR)
print(f"시작 ID 번호: {next_id} (기존 추출 완료된 JSON 파일 {next_id-1}개 확인)")

unique_file_names = sorted(list(set([doc.metadata.get('file_name') for doc in documents if doc.metadata.get('file_name')])))

processed_count = 0
error_count = 0

for file_name in unique_file_names:
    print(f"[{file_name}] 공고문 JSON 추출 시작")
    
    if os.path.exists(os.path.join(EXTRACTED_PDF_DIR, file_name)):
        print(f"[{file_name}] 이미 처리 완료된 파일이므로 건너뜁니다.")
        continue

    extracted_data = extract_structured_json_with_oss(file_name, index)

    if "error" not in extracted_data:
        output_file_name = file_name.replace('.pdf', '.json')
        output_path = os.path.join(OUTPUT_DIR, output_file_name)

        extracted_data['announcement_id'] = next_id
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)

        print(f"[{file_name}] 추출 완료. -> '{output_path}'에 저장됨. (ID: {next_id})")
        processed_count += 1
        
        next_id += 1 
        
        original_path = os.path.join(SOURCE_PDF_DIR, file_name)
        extracted_path = os.path.join(EXTRACTED_PDF_DIR, file_name)
        
        shutil.move(original_path, extracted_path)

    else:
        original_path = os.path.join(SOURCE_PDF_DIR, file_name)
        error_path = os.path.join(ERROR_DIR, file_name)

        print(f"[{file_name}] 오류 발생: {extracted_data['error']}")

        try:
            if os.path.exists(original_path):
                shutil.move(original_path, error_path)
                print(f"    -> 오류 PDF를 '{ERROR_DIR}'로 이동했습니다.")
            else:
                print(f"    -> PDF 파일이 원본 경로({original_path})에 없습니다. 이미 이동되었거나 경로가 잘못되었습니다.")
        except Exception as e:
            print(f"    -> PDF 이동 실패: {e}")

        error_count += 1

try:
    if hasattr(db, "persist"):
        db.persist()
        print(f"\nVector DB가 '{PERSIST_DIR}'에 성공적으로 저장되었습니다.")
    else:
        print("\nVector DB 클라이언트가 persist 메서드를 제공하지 않습니다.")
except Exception as e:
    print(f"\nVector DB 저장 실패: {e}")

print("---")
print(f"총 {processed_count}개의 공고문 요약 JSON이 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")
print(f"오류 발생 공고문 수: {error_count}개")
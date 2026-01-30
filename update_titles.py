import json
import os
from pathlib import Path

def load_titles(titles_path):
    """titles.json 파일을 읽어서 id를 키로 하는 딕셔너리로 변환"""
    with open(titles_path, 'r', encoding='utf-8') as f:
        titles_data = json.load(f)
    
    # id를 키로 하는 딕셔너리 생성
    titles_dict = {item['id']: item for item in titles_data}
    return titles_dict

def update_json_file(json_path, titles_dict):
    """개별 JSON 파일을 업데이트"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # announcement_id가 있는지 확인
        if 'announcement_id' in data and data['announcement_id'] in titles_dict:
            title_info = titles_dict[data['announcement_id']]
            
            # title과 pdf_name 추가/업데이트
            data['title'] = title_info['title']
            data['pdf_name'] = title_info['pdf_name']
            
            # 파일에 다시 저장 (예쁘게 포맷팅)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True, f"Updated: {json_path.name} (ID: {data['announcement_id']})"
        else:
            return False, f"Skipped: {json_path.name} (No matching ID found)"
            
    except Exception as e:
        return False, f"Error processing {json_path.name}: {str(e)}"

def main():
    # 경로 설정
    titles_path = 'titles.json'
    extracted_json_dir = 'extracted_json'
    
    # titles.json 파일 확인
    if not os.path.exists(titles_path):
        print(f"❌ Error: {titles_path} 파일을 찾을 수 없습니다.")
        return
    
    # extracted_json 폴더 확인
    if not os.path.exists(extracted_json_dir):
        print(f"❌ Error: {extracted_json_dir} 폴더를 찾을 수 없습니다.")
        return
    
    # titles.json 로드
    print("📖 titles.json 파일을 읽는 중...")
    titles_dict = load_titles(titles_path)
    print(f"✅ {len(titles_dict)}개의 타이틀 정보를 로드했습니다.\n")
    
    # extracted_json 폴더 내의 모든 JSON 파일 찾기
    json_files = list(Path(extracted_json_dir).glob('*.json'))
    
    if not json_files:
        print(f"❌ {extracted_json_dir} 폴더에 JSON 파일이 없습니다.")
        return
    
    print(f"🔍 {len(json_files)}개의 JSON 파일을 찾았습니다.\n")
    print("=" * 60)
    
    # 각 JSON 파일 업데이트
    success_count = 0
    skip_count = 0
    
    for json_file in sorted(json_files):
        success, message = update_json_file(json_file, titles_dict)
        print(message)
        
        if success:
            success_count += 1
        else:
            skip_count += 1
    
    # 결과 요약
    print("=" * 60)
    print(f"\n✅ 완료!")
    print(f"   - 성공: {success_count}개")
    print(f"   - 건너뜀: {skip_count}개")
    print(f"   - 전체: {len(json_files)}개")

if __name__ == "__main__":
    main()
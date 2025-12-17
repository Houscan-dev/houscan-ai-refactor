<img src="https://github.com/user-attachments/assets/75ec9668-4042-4853-9a51-2398e52cb4e2" width="250" />

### 나에게 맞는 청약 공고와 순위를 한눈에 확인하세요
하우스캔은 **청년을 위한 개인 맞춤형 청약 정보**를 쉽고 빠르게 제공합니다. <br />
AI를 통해 나의 정보를 분석하여 지원 자격에 해당하는지 알 수 있습니다.  <br />
자격이 충족된다면 몇 순위에 해당하는지, 충족되지 않는다면 그 이유를 함께 설명해드립니다.  <br />

### 📸 Demo Video
- <a href='https://youtu.be/sXYb3mKDirA'>시연 영상</a>

### 🔥 Commit Convention
"태그:제목"의 형태이며, : 뒤에만 space가 있음에 유의합니다. ex) Feat: 메인페이지 추가

- `Feat`: 새로운 기능을 추가할 경우
- `Fix`: 버그를 고친 경우
- `Design`: CSS 등 사용자 UI 디자인 변경
- `Docs`: 문서 수정
- `!BREAKING CHANGE`: 커다란 API 변경의 경우 (ex API의 arguments, return 값의 변경, DB 테이블 변경, 급하게 치명적인 버그를 고쳐야 하는 경우)
- `!HOTFIX`: 급하게 치명적인 버그를 고쳐야하는 경우
- `Style`: 코드 포맷 변경, 세미 콜론 누락, 코드 수정이 없는 경우
- `Refactor`: 프로덕션 코드 리팩토링, 새로운 기능이나 버그 수정없이 현재 구현을 개선한 경우
- `Comment`: 필요한 주석 추가 및 변경
- `Test`: 테스트 추가, 테스트 리팩토링(프로덕션 코드 변경 X)
- `Chore`: 빌드 태스트 업데이트, 패키지 매니저를 설정하는 경우(프로덕션 코드 변경 X)
- `Rename`: 파일 혹은 폴더명을 수정하거나 옮기는 작업만인 경우
- `Remove`: 파일을 삭제하는 작업만 수행한 경우

### 🗂️ Directory Structure
```
📦houscan-ai-refactor
 ┣ 📂extracted_json | 추출 및 구조화한 청약 공고 요약 JSON 데이터
 ┣ 📂extracted_pdfs | 처리가 완료되어 분류된 청약 공고문 원본 PDF
 ┣ 📂pdfs | 분석할 청약 공고문 원본 PDF
 ┣ 📜.env
 ┣ 📜.gitignore
 ┣ 📜eligibility_analyzer.py | 사용자별 청약 자격 검증 및 순위 판단
 ┣ 📜extract_subscription.py | 청약 공고문 정보 추출 및 JSON 구조화
 ┗ 📜requirements.txt
```

### 📆 Development Period
2025.03 - 2025.12
<br/>
> 본 프로젝트는 2025년 성신여자대학교 AI융합학부 캡스톤디자인 프로젝트입니다.

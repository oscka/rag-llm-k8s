FROM python:3.9-slim

WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit과 Altair 특정 버전 설치
# RUN pip install --no-cache-dir streamlit==1.12.0 altair==4.2.0

# 애플리케이션 코드 복사
COPY app.py .

# Streamlit 실행
CMD ["streamlit", "run", "app.py"]


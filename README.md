# -ai-_sw
blink_detact 코드 중 
shape_predictor_68_face_landmarks 다운로드 사이트
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

가중치 파일은 학습시키거나 슬랙에 올려둠

실행 아래 단계로 이루어 진다.
1. 파이썬 가상환경 설정
python -m venv myenv
source myenv/bin/activate
pip install flask opencv-python opencv-contrib-python dlib numpy

2.파일 경로 설정
자기가 다운받은 경로로 이동한다.

3.파이썬 구동
python emotion_detect.py 
or
python blink_detect.py


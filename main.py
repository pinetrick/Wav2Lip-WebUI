from flask import Flask, request, jsonify, send_file
import os
import uuid
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
CHECKPOINT_PATH = 'checkpoints/wav2lip_gan.pth'  # 请替换成实际路径

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/inference', methods=['POST'])
def run_inference():
    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Missing audio or video file'}), 400

    # 保存上传的文件
    video_file = request.files['video']
    audio_file = request.files['audio']
    uid = str(uuid.uuid4())

    video_path = os.path.join(UPLOAD_FOLDER, f'{uid}_video.mp4')
    audio_path = os.path.join(UPLOAD_FOLDER, f'{uid}_audio.wav')
    output_path = os.path.join(RESULTS_FOLDER, f'{uid}_result.mp4')

    video_file.save(video_path)
    audio_file.save(audio_path)

    # 调用原始 inference.py
    cmd = [
        'python3', 'inference.py',
        '--checkpoint_path', CHECKPOINT_PATH,
        '--face', video_path,
        '--audio', audio_path,
        '--outfile', output_path
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Inference failed: {e}'}), 500

    if not os.path.exists(output_path):
        return jsonify({'error': 'Output video not found'}), 500

    return send_file(output_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)

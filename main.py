from flask import Flask, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from inference import inference  # 封装你源码里的推理函数
import uuid

app = Flask(__name__)

# 设置允许上传的文件后缀
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav'}

def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext

@app.route('/api/lip_sync', methods=['POST'])
def lip_sync():
    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Video and audio files are required'}), 400

    video_file = request.files['video']
    audio_file = request.files['audio']

    if video_file.filename == '' or audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not (allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS) and allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS)):
        return jsonify({'error': 'Unsupported file type'}), 400

    # 生成唯一ID，避免文件名冲突
    uid = str(uuid.uuid4())

    video_path = os.path.join('temp', f'{uid}_video.{video_file.filename.rsplit(".", 1)[1]}')
    audio_path = os.path.join('temp', f'{uid}_audio.{audio_file.filename.rsplit(".", 1)[1]}')

    # 保存上传文件
    video_file.save(video_path)
    audio_file.save(audio_path)

    # 推理参数，默认可调整
    checkpoint_path = 'checkpoints/wav2lip.pth'
    pads = [0, 10, 0, 0]    # 这里写死或从请求参数读取
    resize_factor = 1
    nosmooth = False

    try:
        # 调用推理函数，生成视频保存到 results/result_voice.mp4
        inference(checkpoint_path, video_path, audio_path, pads, resize_factor, nosmooth)

        output_path = 'results/result_voice.mp4'

        # 返回生成的视频文件
        return send_file(output_path, mimetype='video/mp4', as_attachment=True, download_name='lip_synced_video.mp4')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)

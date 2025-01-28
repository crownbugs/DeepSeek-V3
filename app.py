from flask import Flask, request, render_template_string
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(open('index.html').read())

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    result = subprocess.run(
        ['python3', 'inference/generate.py', '--ckpt-path', 'inference/mock_ckpt.pth', '--config', 'inference/configs/config_16B.json', '--interactive'],
        input=prompt, text=True, capture_output=True
    )
    return f"<h1>Generated Text:</h1><p>{result.stdout}</p>"

if __name__ == '__main__':
    app.run(debug=True)

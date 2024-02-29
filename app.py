from flask import Flask, request, render_template
import model_module  # Import the model module

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    uploaded_file = request.files['file']
    print("File in app.py:", uploaded_file) 
    uploaded_file.save(uploaded_file.filename)
    # Process the image using the model
    extracted_cover = model_module.process_image(uploaded_file)
    # Generate preview and save options
    return render_template('output.html', extracted_cover=extracted_cover)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)

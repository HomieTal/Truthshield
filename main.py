import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils import (setup_database, train_models, load_models, process_news, 
                  test_groq_connection, is_url, fetch_url_content, 
                  extract_text_from_image, allowed_file)

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Tkinter GUI ---
class FakeNewsDetectorGUI:
    def __init__(self, root, sgd_model, pa_model, vectorizer):
        self.root = root
        self.root.title("Fake News Detector")
        self.root.geometry("600x700")
        self.sgd_model = sgd_model
        self.pa_model = pa_model
        self.vectorizer = vectorizer
        
        tk.Label(root, text="Enter News Text, URL, or Select Image:").pack(pady=5)
        self.input_entry = tk.Entry(root, width=60)
        self.input_entry.pack(pady=5)
        
        tk.Label(root, text="Or Select Image:").pack(pady=5)
        self.image_entry = tk.Entry(root, width=60)
        self.image_entry.pack(pady=5)
        tk.Button(root, text="Browse Image", command=self.browse_image).pack(pady=5)
        
        tk.Button(root, text="Analyze", command=self.analyze).pack(pady=10)
        tk.Button(root, text="Retrain Models", command=self.retrain_models).pack(pady=5)
        tk.Button(root, text="Clear Output", command=self.clear_output).pack(pady=5)
        
        tk.Label(root, text="Results:").pack(pady=5)
        self.output_text = scrolledtext.ScrolledText(root, width=70, height=20, wrap=tk.WORD)
        self.output_text.pack(pady=5)
        
        self.output_text.insert(tk.END, "🔍 Fake News Detection System\n")
        self.output_text.insert(tk.END, "============================\n")
        self.output_text.insert(tk.END, "Enter news text, a URL, or select an image to analyze.\n")
        self.output_text.see(tk.END)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_entry.delete(0, tk.END)
            self.image_entry.insert(0, file_path)

    def analyze(self):
        input_text = self.input_entry.get().strip()
        image_path = self.image_entry.get().strip()
        
        if not (input_text or image_path):
            messagebox.showerror("Input Error", "Please provide news text, a URL, or an image.")
            return
        
        if image_path:
            self.output_text.insert(tk.END, f"\n🖼️ Extracting text from image: {image_path}\n")
            self.output_text.see(tk.END)
            news_text = extract_text_from_image(image_path)
            if news_text:
                self.output_text.insert(tk.END, f"\n📝 Extracted Text: {news_text}\n")
                self.sgd_model, self.pa_model, self.vectorizer = process_news(
                    news_text, self.sgd_model, self.pa_model, self.vectorizer, self.output_text
                )
            else:
                self.output_text.insert(tk.END, "⚠️ Cannot proceed with analysis due to image processing failure.\n")
                self.output_text.see(tk.END)
        elif input_text:
            if is_url(input_text):
                self.output_text.insert(tk.END, f"\n🌐 Fetching content from URL: {input_text}\n")
                self.output_text.see(tk.END)
                news_text = fetch_url_content(input_text)
                if news_text:
                    self.sgd_model, self.pa_model, self.vectorizer = process_news(
                        news_text, self.sgd_model, self.pa_model, self.vectorizer, self.output_text
                    )
                else:
                    self.output_text.insert(tk.END, "⚠️ Cannot proceed with analysis due to URL fetch failure.\n")
                    self.output_text.see(tk.END)
            else:
                self.output_text.insert(tk.END, f"\n📝 Processing news text: {input_text}\n")
                self.output_text.see(tk.END)
                self.sgd_model, self.pa_model, self.vectorizer = process_news(
                    input_text, self.sgd_model, self.pa_model, self.vectorizer, self.output_text
                )

    def retrain_models(self):
        if messagebox.askyesno("Retrain Models", "Retrain models from scratch? This may take a while."):
            self.output_text.insert(tk.END, "\n🧠 Training new models from scratch...\n")
            self.output_text.see(tk.END)
            self.sgd_model, self.pa_model, self.vectorizer = train_models()
            self.output_text.insert(tk.END, "✅ Models retrained successfully.\n")
            self.output_text.see(tk.END)

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "🔍 Fake News Detection System\n")
        self.output_text.insert(tk.END, "============================\n")
        self.output_text.insert(tk.END, "Enter news text, a URL, or select an image to analyze.\n")
        self.output_text.see(tk.END)

# --- Flask API Endpoint ---
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        class StringOutput:
            def __init__(self):
                self.output = []
            def insert(self, _, text):
                self.output.append(text)
            def see(self, _):
                pass
        output_widget = StringOutput()

        sgd_model, pa_model, vectorizer = load_models()

        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                news_text = extract_text_from_image(image_path)
                if not news_text:
                    return jsonify({'error': 'Failed to extract text from image'}), 400
            else:
                return jsonify({'error': 'Invalid image file'}), 400
        elif 'url' in request.form and request.form['url']:
            url = request.form['url']
            if is_url(url):
                news_text = fetch_url_content(url)
                if not news_text:
                    return jsonify({'error': 'Failed to fetch content from URL'}), 400
            else:
                return jsonify({'error': 'Invalid URL'}), 400
        elif 'text' in request.form and request.form['text']:
            news_text = request.form['text']
        else:
            return jsonify({'error': 'No valid input provided'}), 400

        sgd_model, pa_model, vectorizer = process_news(news_text, sgd_model, pa_model, vectorizer, output_widget)

        return jsonify({
            'results': output_widget.output,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Fake News Detection System")
    parser.add_argument("--train", action="store_true", help="Train new models")
    parser.add_argument("--force-retrain", action="store_true", help="Force training new models")
    parser.add_argument("--input", type=str, help="News text or URL to analyze")
    parser.add_argument("--image", type=str, help="Path to image containing news text")
    parser.add_argument("--no-db", action="store_true", help="Skip database connection attempt")
    parser.add_argument("--test-groq", action="store_true", help="Test Groq API connection")
    parser.add_argument("--api", action="store_true", help="Run as API server")
    args = parser.parse_args()

    print("\n🔍 Fake News Detection System")
    print("============================")

    if not args.no_db:
        setup_database()

    if args.test_groq:
        test_groq_connection()

    if args.api:
        print("\n🚀 Starting Flask API server...")
        app.run(host='0.0.0.0', port=4000, debug=True)
    else:
        if args.train or args.force_retrain:
            print("\n🧠 Training new models from scratch...")
            sgd_model, pa_model, vectorizer = train_models()
        else:
            print("\n🔄 Loading existing models...")
            sgd_model, pa_model, vectorizer = load_models()

        if args.input or args.image:
            output_widget = scrolledtext.ScrolledText()
            current_models = (sgd_model, pa_model, vectorizer)
            if args.input:
                if is_url(args.input):
                    print(f"\n🌐 Fetching content from URL: {args.input}")
                    news_text = fetch_url_content(args.input)
                    if news_text:
                        sgd_model, pa_model, vectorizer = process_news(news_text, *current_models, output_widget)
                    else:
                        print("⚠️ Cannot proceed with analysis due to URL fetch failure.")
                else:
                    print(f"\n📝 Processing news text: {args.input}")
                    sgd_model, pa_model, vectorizer = process_news(args.input, *current_models, output_widget)
            elif args.image:
                print(f"\n🖼️ Extracting text from image: {args.image}")
                news_text = extract_text_from_image(args.image)
                if news_text:
                    print(f"\n📝 Extracted Text: {news_text}")
                    sgd_model, pa_model, vectorizer = process_news(news_text, *current_models, output_widget)
                else:
                    print("⚠️ Cannot proceed with analysis due to image processing failure.")
        else:
            root = tk.Tk()
            app = FakeNewsDetectorGUI(root, sgd_model, pa_model, vectorizer)
            root.mainloop()

if __name__ == "__main__":
    groq_available = test_groq_connection()
    if not groq_available:
        print("\n⚠️ Groq API is not available or key is invalid.")
        print("The system will use rule-based analysis for AI features.")
    
    main()
